$(document).ready(function(){
  $("#video-active").on(
    "timeupdate",
    function(event){
      onTrackedVideoFrame(this.currentTime, this.duration);
    });
});

function onTrackedVideoFrame(currentTime, duration){
    $("#current").text(currentTime); //Change #current to currentTime
    $("#duration").text(duration)
}
