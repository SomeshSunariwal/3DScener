"use strict";

function createLocalRuntime(node, log) {
  const path = node.require("path");
  const fs = node.require("fs");
  const os = node.require("os");
  const { Buffer } = node.require("buffer");
  const { fileURLToPath } = node.require("url");
  const { spawn } = node.require("child_process");
  const { randomBytes } = node.require("crypto");
  const env = node.process.env;
  const root = path.dirname(fileURLToPath(window.location.href));
  const data = path.join(env.LOCALAPPDATA, "3DScener");
  const worker = path.join(root, "worker", "scener.py");
  const models = JSON.parse(fs.readFileSync(path.join(root, "worker", "models.json"), "utf8"));
  const venv = path.join(data, "runtime");
  const runtime = path.join(venv, "Scripts", "python.exe");

  function run(label, command, args, onEvent, { input, signal, jobId } = {}) {
    return new Promise((resolve, reject) => {
      if (signal && signal.aborted) {
        reject(new Error("Processing stopped."));
        return;
      }
      log("Starting " + label);
      const child = spawn(command, args, {
        windowsHide: true, cwd: root,
        env: jobId ? { ...env, SCENER_JOB_ID: jobId } : env,
      });
      const abort = () => child.kill();
      if (signal) signal.addEventListener("abort", abort, { once: true });
      let processError;
      if (input !== undefined) {
        child.stdin.on("error", (error) => {
          if (error.code !== "EPIPE") processError = error;
        });
        child.stdin.end(input);
      }
      let stdout = "";
      let stderr = "";
      function streamLines(stream, receive, collect) {
        let pending = "";
        stream.setEncoding("utf8");
        stream.on("data", (chunk) => {
          if (collect) collect(chunk);
          const lines = (pending + chunk).split(/[\r\n]+/);
          pending = lines.pop();
          for (const line of lines) {
            if (line.trim()) receive(line);
          }
        });
        return () => {
          if (pending.trim()) receive(pending);
        };
      }
      const flushOutput = streamLines(child.stdout, (line) => {
        if (onEvent) {
          let event;
          try { event = JSON.parse(line); } catch (_) { /* Plain Python output. */ }
          if (event && (event.result !== undefined || event.error || event.progress)) {
            onEvent(event);
            return;
          }
        }
        log(line);
      }, onEvent ? null : (chunk) => { stdout = (stdout + chunk).slice(-4000); });
      const flushErrors = streamLines(child.stderr, (line) => log("stderr: " + line),
        (chunk) => { stderr = (stderr + chunk).slice(-4000); });
      child.on("error", (error) => {
        log(label + ": " + error.message);
        processError = error;
      });
      child.on("close", (code) => {
        if (signal) signal.removeEventListener("abort", abort);
        flushOutput();
        flushErrors();
        if (signal && signal.aborted) {
          log("Stopped " + label);
          reject(new Error("Processing stopped."));
        } else if (processError) {
          reject(processError);
        } else if (code === 0) {
          log("Finished " + label);
          resolve(stdout.trim());
        } else {
          log(label + " stopped (exit " + code + ")");
          reject(new Error(stderr.trim().split(/\r?\n/).pop() || "The local process stopped (" + code + ")."));
        }
      });
    });
  }

  async function workerCall(python, args, onProgress, payload, signal) {
    let result;
    let failure;
    let lastProgress;
    const [action, key] = args;
    const jobId = signal && action === "process" ? randomBytes(16).toString("hex") : undefined;
    const label = {
      check: "runtime and model checks",
      preview: "image preview",
      mask: "object mask with " + (models[args[3]] || models["mobile-sam"]).label,
      download: "download of " + (models[key] ? models[key].label : key),
      delete: "deletion of " + (models[key] ? models[key].label : key),
      process: "image processing with " + (models[key] ? models[key].label : key),
    }[action] || action;
    try {
      await run(label, python[0], [
        ...python.slice(1), "-B", "-X", "utf8", "-u", worker, args[0], data, ...args.slice(1),
      ], (event) => {
        if (signal && signal.aborted) return;
        if (event.result !== undefined) {
          result = event.result;
          if (action === "check" && result && result.runtimeError) log("Runtime check: " + result.runtimeError);
        }
        if (event.error) {
          failure = event.error;
          log("Error: " + failure);
        }
        if (event.progress) {
          const message = event.total ? event.progress + " (" + Math.floor(event.received / event.total * 100) + "%)" :
            event.received ? event.progress + " (" + Math.floor(event.received / 5000000) * 5 + " MB received)" : event.progress;
          if (message !== lastProgress) log(message);
          lastProgress = message;
          if (onProgress) onProgress(event);
        }
      }, { input: payload === undefined ? undefined : JSON.stringify(payload), signal, jobId });
    } catch (error) {
      if (jobId) {
        const job = path.join(data, "scenes", jobId);
        await fs.promises.rm(job, { recursive: true, force: true }).catch((failure) => log(failure.message));
      }
      throw new Error(failure || error.message);
    }
    if (result === undefined) {
      log("Error: the local worker returned no result.");
      throw new Error("The local worker returned no result.");
    }
    return result;
  }

  async function findPython(explicit, remembered, signal) {
    const candidates = explicit ? [[explicit]] : [
      ...(remembered ? [[remembered]] : []),
      ...["313", "312", "311"].map((version) => [
        path.join(env.LOCALAPPDATA, "Programs", "Python", "Python" + version, "python.exe"),
      ]),
      ["py", "-3.13"], ["py", "-3.12"], ["py", "-3.11"], ["python"],
    ];
    for (const candidate of candidates) {
      try {
        const result = await run("Python check (" + path.basename(candidate[0]) + ")", candidate[0], [...candidate.slice(1), "-c",
          "import platform, sys; print('.'.join(map(str, sys.version_info[:3]))) if " +
          "sys.implementation.name == 'cpython' and (3,11) <= sys.version_info[:2] <= (3,13) " +
          "and platform.machine().upper() == 'AMD64' else sys.exit(1)",
        ], undefined, { signal });
        return { command: candidate, version: result };
      } catch (error) {
        if (signal && signal.aborted) throw error;
        // Continue through the supported locations and launcher versions.
      }
    }
    return null;
  }

  async function install(python) {
    const [command, ...args] = python;
    await run("virtual environment setup", command, [...args, "-u", "-m", "venv", venv]);
    await run("runtime package installation", runtime, ["-u", "-m", "pip", "install", "--disable-pip-version-check",
      "--requirement", path.join(root, "worker", "requirements.txt")]);
  }

  async function installMat() {
    const pip = ["-u", "-m", "pip", "install", "--disable-pip-version-check"];
    await run("MAT CPU packages", runtime, [...pip, "--index-url", "https://download.pytorch.org/whl/cpu",
      "torch==2.10.0", "torchvision==0.25.0"]);
    await run("MAT model support", runtime, [...pip, "spandrel==0.4.2", "spandrel-extra-arches==0.2.0"]);
  }

  async function withImagePath(file, action) {
    if (typeof file.path === "string" && path.isAbsolute(file.path) && fs.existsSync(file.path)) {
      return action(file.path);
    }
    const bytes = Buffer.from(await file.arrayBuffer());
    const directory = await fs.promises.mkdtemp(path.join(os.tmpdir(), "3DScener-"));
    const filename = path.join(directory, path.basename(file.name));
    try {
      await fs.promises.writeFile(filename, bytes);
      return await action(filename);
    } finally {
      await fs.promises.unlink(filename).catch(() => {});
      await fs.promises.rmdir(directory).catch(() => {});
    }
  }

  return {
    models, findPython, install, installMat, withImagePath,
    workerCall, runtime, hasRuntime: () => fs.existsSync(runtime),
  };
}
