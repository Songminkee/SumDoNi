$(document).ready(function(){
    imageTempl = document.getElementById("image-template");
    videoTempl = document.getElementById("video-template");
    empty = document.getElementById("empty");
    file_button = document.getElementById("button");

    // use to store pre selected files
    let FILES = {};

    // check if file is of type image and prepend the initialied
    // template to the target element
    function addFile(target, file) {
        const isImage = file.type.match("image.*"),
        objectURL = URL.createObjectURL(file);

        const clone = isImage
        ? imageTempl.content.cloneNode(true)
        : videoTempl.content.cloneNode(true);

        if(isImage) {
        Object.assign(clone.querySelector("img"), {
            src: objectURL,
            alt: file.name
        });
        }
        else {
        Object.assign(clone.querySelector("source"), {
            src: objectURL,
            alt: file.name
        });
        }

        empty.classList.add("hidden");
        target.prepend(clone);

        FILES[objectURL] = file;

        file_button.remove()
    }

    const gallery = document.getElementById("gallery"),
    overlay = document.getElementById("overlay");

    // click the hidden input of type file if the visible button is clicked
    // and capture the selected files
    const hidden = document.getElementById("hidden-input");
    document.getElementById("button").onclick = () => hidden.click();
    hidden.onchange = (e) => {
        for (const file of e.target.files) {
            addFile(gallery, file);
        }
    };

});

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

    alert("현재 사진으로 얼굴인식을 시작합니다.");

    document.getElementById('form').submit();
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