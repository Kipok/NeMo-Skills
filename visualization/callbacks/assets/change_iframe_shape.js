window.addEventListener('message', function(event) {
    if (event.data && event.data.frameHeight && event.data.frameId) {
        var iframe = document.getElementById(event.data.frameId);
        if (iframe) {
            iframe.style.height = event.data.frameHeight + 'px';
        }
    }
}, false);
