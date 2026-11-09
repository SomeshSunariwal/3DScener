// Called by the CEP panel. The return value is displayed in the panel.
function create3DScenerScene(scene) {
    var project = app.project;
    if (!project) return "Open an After Effects project first.";

    var files = {
        background: new File(scene.background),
        depth: new File(scene.depth)
    };
    var objects = scene.layers || [{ name: "Foreground", file: scene.foreground, x: 0, y: 0, nearness: 1 }];
    if (!files.background.exists || !files.depth.exists) {
        return "A scene file is missing. Process the image again.";
    }
    var objectFiles = [];
    for (var index = 0; index < objects.length; index++) {
        var file = new File(objects[index].file);
        if (!file.exists) return "An object layer file is missing. Process the image again.";
        objectFiles.push(file);
    }

    var created = [];
    app.beginUndoGroup("Create 3DScener scene");
    try {
        var folder = project.items.addFolder("3DScener - " + scene.name);
        created.push(folder);

        function importStill(file, name) {
            var item = project.importFile(new ImportOptions(file));
            created.push(item);
            item.name = name;
            item.parentFolder = folder;
            return item;
        }

        var background = importStill(files.background, "Background Fill");
        var footage = [];
        for (var index = 0; index < objects.length; index++) {
            var item = importStill(objectFiles[index], objects[index].name);
            item.mainSource.alphaMode = AlphaMode.STRAIGHT;
            footage.push(item);
        }
        var depth = importStill(files.depth, "Depth Map");
        var width = scene.width;
        var height = scene.height;
        var center = [width / 2, height / 2];
        var comp = project.items.addComp(scene.name + " - Parallax", width, height, 1, 5, 30);
        created.push(comp);
        comp.parentFolder = folder;

        var camera = comp.layers.addCamera("3DScener Camera", center);
        camera.autoOrient = AutoOrientType.NO_AUTO_ORIENT;
        var zoom = camera.property("ADBE Camera Options Group").property("ADBE Camera Zoom").value;
        var cameraPosition = camera.property("ADBE Transform Group").property("ADBE Position");
        cameraPosition.setValueAtTime(0, [center[0], center[1], -zoom]);
        cameraPosition.setValueAtTime(comp.duration - comp.frameDuration, [center[0] + width * 0.04, center[1], -zoom]);

        function addPlane(item, z, object) {
            var layer = comp.layers.add(item);
            layer.threeDLayer = true;
            var transform = layer.property("ADBE Transform Group");
            // Compensate for perspective so the cutout matches the photo at frame zero.
            var scale = (zoom + z) / zoom;
            var x = object ? center[0] + (object.x + item.width / 2 - center[0]) * scale : center[0];
            var y = object ? center[1] + (object.y + item.height / 2 - center[1]) * scale : center[1];
            transform.property("ADBE Position").setValue([x, y, z]);
            transform.property("ADBE Scale").setValue([100 * scale, 100 * scale, 100 * scale]);
            return layer;
        }

        var depthLayer = comp.layers.add(depth);
        depthLayer.enabled = false;
        addPlane(background, zoom * 0.18);
        for (var index = 0; index < objects.length; index++) {
            addPlane(footage[index], zoom * (0.14 - 0.26 * objects[index].nearness), objects[index]);
        }
        camera.moveToBeginning();
        comp.openInViewer();
        camera.selected = true;
        return "OK";
    } catch (error) {
        for (var index = created.length - 1; index >= 0; index--) {
            try { created[index].remove(); } catch (ignored) {}
        }
        return error.toString();
    } finally {
        app.endUndoGroup();
    }
}
