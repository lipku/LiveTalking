var pc = null;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        document.getElementById('sessionid').value = answer.sessionid
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        alert(e);
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);
    
    pc.onconnectionstatechange = function() {
        console.log('WebRTC连接状态:', pc.connectionState);
    };

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            const videoElement = document.getElementById('video');
            if (videoElement) {
                videoElement.srcObject = evt.streams[0];
            } else {
                console.error('Video element not found');
            }
        } else {
            const audioElement = document.getElementById('audio');
            if (audioElement) {
                audioElement.srcObject = evt.streams[0];
            } else {
                console.error('Audio element not found');
            }
        }
    });

    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}

window.onunload = function(event) {
    // 在这里执行你想要的操作
    setTimeout(() => {
        pc.close();
    }, 500);
};

window.onbeforeunload = function (e) {
    setTimeout(() => {
        pc.close();
    }, 500);
    e = e || window.event
    // 兼容IE8和Firefox 4之前的版本
    if (e) {
      e.returnValue = '关闭提示'
    }
    // Chrome, Safari, Firefox 4+, Opera 12+ , IE 9+
    return '关闭提示'
}
