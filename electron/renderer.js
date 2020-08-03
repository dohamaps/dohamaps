
const { ipcRenderer } = require("electron");
const path = require("path");

let loadPath = document.getElementById("load-path");
let loadNumClips = document.getElementById("load-num-clips");
let loadSubmit = document.getElementById("load-submit");
let loadSpinner = document.getElementById("load-spinner");
let loadBody = document.getElementById("load-body");
let loadPrint = document.getElementById("load-print");

let trainEpochs = document.getElementById("train-epochs");
let trainBatchSize = document.getElementById("train-batch-size");
let trainSubmit = document.getElementById("train-submit");
let trainSpinner = document.getElementById("train-spinner");
let trainBody = document.getElementById("train-body");
let trainPrint = document.getElementById("train-print");

let testSteps = document.getElementById("test-steps");
let testBatchSize = document.getElementById("test-batch-size");
let testSubmit = document.getElementById("test-submit");
let testSpinner = document.getElementById("test-spinner");
let testBody = document.getElementById("test-body");
let testPrint = document.getElementById("test-print");

let predImgDisplay = document.getElementById("pred-img-display");
let predDownload = document.getElementById("pred-download");
let predSubmit = document.getElementById("pred-submit");
let predSubmitOriginal = document.getElementById("pred-submit-original");
let predSpinner = document.getElementById("pred-spinner");
let predBody = document.getElementById("pred-body");
let predPrint = document.getElementById("pred-print");

function disableSubmits() {
    loadSubmit.classList.add("disabled");
    trainSubmit.classList.add("disabled");
    testSubmit.classList.add("disabled");
    predSubmit.classList.add("disabled");
    predSubmitOriginal.classList.add("disabled");
}

function enableSubmits() {
    loadSubmit.classList.remove("disabled");
    trainSubmit.classList.remove("disabled");
    testSubmit.classList.remove("disabled");
    predSubmit.classList.remove("disabled");
    predSubmitOriginal.classList.remove("disabled");
}

function showSpinners() {
    loadSpinner.style.display = "flex";
    trainSpinner.style.display = "flex";
    testSpinner.style.display = "flex";
    predSpinner.style.display = "flex";
}

function hideSpinners() {
    loadSpinner.style.display = "none";
    trainSpinner.style.display = "none";
    testSpinner.style.display = "none";
    predSpinner.style.display = "none";
}

function showBodies() {
    loadBody.style.display = "block";
    trainBody.style.display = "block";
    testBody.style.display = "block";
    predBody.style.display = "block";
}

function hideBodies() {
    loadBody.style.display = "none";
    trainBody.style.display = "none";
    testBody.style.display = "none";
    predBody.style.display = "none";
}

function enableLoading() {
    disableSubmits();
    showSpinners();
    hideBodies();
}

function disableLoading() {
    enableSubmits();
    hideSpinners();
    showBodies();
}

loadSubmit.addEventListener("click", function () {
    if (!loadSubmit.classList.contains("disabled")) {
        enableLoading();
        loadPrint.style.display = "block";
        let __path = path.parse(loadPath.files[0].path).dir;
        let __numClips = loadNumClips.value;
        loadPath.files = null;
        loadNumClips.value = "";
        ipcRenderer.send("load", [ __numClips, __path ]);
    }
});

trainSubmit.addEventListener("click", function () {
    if (!trainSubmit.classList.contains("disabled")) {
        enableLoading();
        trainPrint.style.display = "block";
        let __epochs = trainEpochs.value;
        let __batchSize = trainBatchSize.value;
        trainEpochs.value = "";
        trainBatchSize.value = "";
        ipcRenderer.send("train", [ __epochs, __batchSize ]);
    }
});

testSubmit.addEventListener("click", function () {
    if (!testSubmit.classList.contains("disabled")) {
        enableLoading();
        testPrint.style.display = "block";
        let __steps = testSteps.value;
        let __batchSize = testBatchSize.value;
        testSteps.value = "";
        testBatchSize.value = "";
        ipcRenderer.send("test", [ __steps, __batchSize ]);
    }
});

predSubmit.addEventListener("click", function () {
    if (!predSubmit.classList.contains("disabled")) {
        enableLoading();
        predPrint.style.display = "block";
        ipcRenderer.send("pred", [ false ]);
    }
});

predSubmitOriginal.addEventListener("click", function () {
    if (!predSubmitOriginal.classList.contains("disabled")) {
        enableLoading();
        predPrint.style.display = "block";
        ipcRenderer.send("pred", [ true ]);
    }
});

predDownload.addEventListener("click", function() {
    if (!predDownload.classList.contains("disabled")) {
        ipcRenderer.send("download", [ "../save/pred/pred.png" ]);
    }
});

ipcRenderer.on("load", function (event, arg) {
    disableLoading();
    loadPrint.style.display = "none";
});

ipcRenderer.on("train", function (event, arg) {
    disableLoading();
    trainPrint.style.display = "none";
});

ipcRenderer.on("test", function (event, arg) {
    disableLoading();
    testPrint.style.display = "none";
});

ipcRenderer.on("pred", function (event, arg) {
    disableLoading();
    predPrint.style.display = "none";
    let predImg = document.createElement("img");
    predImg.src = "../save/pred/pred.png";
    predImg.height = 400;
    predImg.classList.add("pred-img");
    predImgDisplay.classList.add("pred-img-active");
    if (!predImgDisplay.classList.contains("__active")) {
        predImgDisplay.appendChild(predImg);
        predImgDisplay.classList.add("__active");
    }
    setTimeout(function() {
        predImg.style.opacity = "100%";
        predDownload.classList.remove("disabled");
    }, 2000);
});

ipcRenderer.on("load-print", function (event, arg) {
    loadPrint.innerHTML = arg[0];
});

ipcRenderer.on("train-print", function (event, arg) {
    trainPrint.innerHTML = arg[0];
});

ipcRenderer.on("test-print", function (event, arg) {
    testPrint.innerHTML = arg[0];
});

ipcRenderer.on("pred-print", function (event, arg) {
    predPrint.innerHTML = arg[0];
});
