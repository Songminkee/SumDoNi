$(document).ready(function(){
    fps = document.getElementById("fps").value
    $("#video-active").on(
    "timeupdate",
    function(event){
      onTrackedVideoTime(this.currentTime, this.duration);
      onTrackedVideoFrame(parseInt((this.currentTime * parseFloat(fps)).toString()), this.duration * parseFloat(fps));
    });
});

function onTrackedVideoTime(currentTime, duration){
    $("#currentTime").text(currentTime); //Change #current to currentTime
    $("#duration").text(duration)
}

function onTrackedVideoFrame(currentFrame, duration){
    $("#currentFrame").text(currentFrame); //Change #current to currentTime
    $("#durationFrame").text(duration)
}

function getFrameFromVideo(){
    video = document.getElementById("video-active");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    // draw the video at that frame
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    // convert it to a usable data URL
    const dataURL = canvas.toDataURL();

    document.getElementById("img").value = dataURL;
    document.getElementById("frame").value = document.getElementById("currentFrame").innerText
}

window.addEventListener('keypress', function (evt) {
    if (video.paused) { //or you can force it to pause here
        if (evt.keyCode === 37) { //left arrow
            //one frame back
            video.currentTime = Math.max(0, video.currentTime - frameTime);
        } else if (evt.keyCode === 39) { //right arrow
            //one frame forward
            //Don't go past the end, otherwise you may get an error
            video.currentTime = Math.min(video.duration, video.currentTime + frameTime);
        }
    }
});