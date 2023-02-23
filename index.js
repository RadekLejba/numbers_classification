
const canvas = document.getElementById("myCanvas");
const clear = document.getElementById("clear");
const identify = document.getElementById("identify");
const context = canvas.getContext("2d");
const download = document.getElementById('download');
const train = document.getElementById('train');

context.strokeStyle = "#000";
context.lineWidth = 4;
context.fillStyle = "white";
context.fillRect(0, 0, canvas.width, canvas.height);

let x = 0;
let y = 0;
let isDrawing = false;

canvas.addEventListener("mousedown", e => {
    isDrawing = true;
    x = e.offsetX;
    y = e.offsetY;
});

canvas.addEventListener("mousemove", e => {
    if (isDrawing) {
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(e.offsetX, e.offsetY);
        context.stroke();
        x = e.offsetX;
        y = e.offsetY;
    }
});

canvas.addEventListener("mouseup", e => {
    isDrawing = false;
});

canvas.addEventListener("mouseout", e => {
    isDrawing = false;
});

clear.addEventListener("click", e =>{
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillRect(0, 0, canvas.width, canvas.height);
});

download.addEventListener('click', function (e) {
    const link = document.createElement('a');
    link.download = 'download.png';
    link.href = canvas.toDataURL("image/bmp");
    link.click();
    link.delete;
});


identify.addEventListener("click", e =>{
    const imageData = canvas.toDataURL("image/bmp");
    $.ajax({
        type: "POST",
        contentType: "application/json",
        dataType: "json",
        url: "http://localhost:80/identify_image",
        data: JSON.stringify({ 
            image: imageData
        }),
        success: function(data) {
            $('#message').text("Image classified as: " + data.label);
        }
    });
});

train.addEventListener("click", e =>{
    $('#message').text("Training started, please wait");
    $.ajax({
        type: "GET",
        url: "http://localhost:80/train",
        success: function(data) {
            $('#message').text("Model trained successfuly");
        }
    });
});