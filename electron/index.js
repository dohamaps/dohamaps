
const { app, BrowserWindow, ipcMain, nativeImage } = require("electron");
const path = require("path");
const { PythonShell } = require("python-shell");
const fs = require("fs-extra");
const homedir = require("os").homedir();
// require("update-electron-app")();

var image = nativeImage.createFromPath(path.join(__dirname, "icon.png"));
image.setTemplateImage(true);

app.dock.setIcon(image);

const pythonPath = path.join(__dirname, "../venv/bin/python3");
const scriptPath = path.join(__dirname, "../");
const scriptName = "main.py";

const __debug__ = true;

function __sysprint__(value) {
    if (__debug__) console.log(value);
}

let loadOptions = {
    mode: "text",
    pythonPath: pythonPath,
    pythonOptions: ["-u"],
    scriptPath: scriptPath,
    args: [ "--process=", "--data-dir=", "--log-html" ]
};

let trainOptions = {
    mode: "text",
    pythonPath: pythonPath,
    pythonOptions: ["-u"],
    scriptPath: scriptPath,
    args: [ "--train=", "--batch-size=", "--log-html" ]
};

let testOptions = {
    mode: "text",
    pythonPath: pythonPath,
    pythonOptions: ["-u"],
    scriptPath: scriptPath,
    args: [ "--test=", "--batch-size=", "--log-html" ]
};

let predOptions = {
    mode: "text",
    pythonPath: pythonPath,
    pythonOptions: ["-u"],
    scriptPath: scriptPath,
    args: [ "--predict", "--log-html" ]
};

if (require("electron-squirrel-startup")) {
    app.quit();
}

function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        minWidth: 800,
        minHeight: 600,
        titleBarStyle: "hiddenInset",
        frame: false,
        icon: image,
        webPreferences: {
            nodeIntegration: true
        }
    });

    mainWindow.loadFile(path.join(__dirname, "index.html"));

    ipcMain.on("load", function (event, arg) {
        loadOptions.args[0] += arg[0];
        loadOptions.args[1] += arg[1];
        let script = PythonShell.run(scriptName, loadOptions);
        script.on("message", function (message) {
            event.reply("load-print", [ message ]);
        });
        script.end(function (err, code, signal) {
            if (err) console.log(err);
            event.reply("load", [  ]);
        });
    });

    ipcMain.on("train", function (event, arg) {
        trainOptions.args[0] += arg[0];
        trainOptions.args[1] += arg[1];
        let script = PythonShell.run(scriptName, trainOptions);
        script.on("message", function (message) {
            event.reply("train-print", [ message ]);
        });
        script.end(function (err, code, signal) {
            if (err) console.log(err);
            event.reply("train", [  ]);
        });
    });

    ipcMain.on("test", function (event, arg) {
        testOptions.args[0] += arg[0];
        testOptions.args[1] += arg[1];
        let script = PythonShell.run(scriptName, testOptions);
        script.on("message", function (message) {
            event.reply("test-print", [ message ]);
        });
        script.end(function (err, code, signal) {
            if (err) console.log(err);
            event.reply("test", [  ]);
        });
    });

    ipcMain.on("pred", function (event, arg) {
        if (arg[0]) {
            predOptions.args.push("--weights-dir=");
            predOptions.args[2] += path.join(__dirname, "../checkpt_original");
        }
        let script = PythonShell.run(scriptName, predOptions);
        script.on("message", function (message) {
            event.reply("pred-print", [ message ]);
        });
        script.end(function (err, code, signal) {
            if (err) console.log(err);
            event.reply("pred", [  ]);
        });
    });

    ipcMain.on("download", function (event, arg) {
        fs.copy(path.join(__dirname, arg[0]),
                path.join(homedir, "Downloads/pred.png"),
                function (err) { if (err) return console.error(err); }
        );
    });
};

app.on("ready", createWindow);

app.on("window-all-closed", function () {
    if (process.platform !== "darwin") {
        app.quit();
    }
});

app.on("activate", function () {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
