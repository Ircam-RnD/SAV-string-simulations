import { html, render } from 'lit';

import '@ircam/sc-components';
import resumeAudioContext from './resume-audio-context.js';


const audioContext = new AudioContext({ sampleRate: 44100 });
await resumeAudioContext(audioContext);
await audioContext.audioWorklet.addModule('./src/StringProcessor.js');

let impulseWidth = 1 / 1000;


// Parameters
let options = {};
options.processorOptions = {};
options.processorOptions.t60_0 = 12;
options.processorOptions.t60_1000 = 8;
options.processorOptions.f0 = 150;
options.processorOptions.beta = 0.001;

let stringNode = new AudioWorkletNode(audioContext, 'string-processor', options);
const inGainNode = new GainNode(audioContext);
inGainNode.gain.value = 0.8;
const outGainNode = new GainNode(audioContext);
inGainNode.connect(stringNode).connect(outGainNode).connect(audioContext.destination);


function struckSource() {
  const N = impulseWidth * audioContext.sampleRate;
  const buffer = audioContext.createBuffer(1, N, audioContext.sampleRate);
  // Fill with half a sine wave
  for (let i = 0; i < N; i++) {
    buffer.getChannelData(0)[i] = Math.sin(i / N * Math.PI);
  }
  const bufferSource = audioContext.createBufferSource();
  bufferSource.buffer = buffer;
  bufferSource.connect(inGainNode);
  bufferSource.start();
}

function pluckSource() {
  const N = impulseWidth * audioContext.sampleRate;
  const buffer = audioContext.createBuffer(1, N, audioContext.sampleRate);
  // Fill with half a sine wave
  for (let i = 0; i < N; i++) {
    buffer.getChannelData(0)[i] = Math.sin(i / (2 * N) * Math.PI);
  }
  const bufferSource = audioContext.createBufferSource();
  bufferSource.buffer = buffer;
  bufferSource.connect(inGainNode);
  bufferSource.start();
}

function updateParameters() {
  stringNode.port.postMessage(options);
}

render(html`
  <h1>Interactive simulation</h1>
  <div style="position: absolute; top: 20px; right: 20px;">
    <sc-icon type="github" href="https://github.com/thomas-risse/SAV-string-simulations"></sc-icon>
  </div>
  <div>
  <div style="padding: 4px 0;"></div>
      <sc-button @release=${updateParameters}>Update parameters </sc-button>
      <sc-text> Fundamental frequency </sc-text>
      <sc-number
        min = 20
        max = 10000
        value=${options.processorOptions.f0}
        @input=${function(e){
          options.processorOptions.f0 = e.detail.value;
        }}
      ></sc-number> 
      <sc-text> Inharmonicity </sc-text>
      <sc-number
        min = 1e-5
        max = 1
        value=${options.processorOptions.beta}
        @input=${function(e){
          options.processorOptions.beta = e.detail.value;
        }}
      ></sc-number> 
      <sc-text> Decay time 0 Hz </sc-text>
      <sc-number
        min = 0.01
        max = 1e10
        value=${options.processorOptions.t60_0}
        @input=${function(e){
          options.processorOptions.t60_0 = e.detail.value;
        }}
      ></sc-number> 
      <sc-text> Decay time 1000 Hz </sc-text>
      <sc-number
        min = 0.01
        max = 1e10
        value=${options.processorOptions.t60_1000}
        @input=${function(e){
          options.processorOptions.t60_1000 = e.detail.value;
        }}
      ></sc-number> 
    </div>
    <div style="padding: 4px 0;"></div>
      <sc-button @release=${struckSource}>Impulse (struck) </sc-button>
      <sc-button @release=${pluckSource}>Impulse (pluck) </sc-button>
    </div>
    <div style="padding: 4px 0;"></div>
      <sc-text> Input gain </sc-text>
      <sc-slider
        min = "0"
        max="2"
        number-box
        value=${inGainNode.gain.value}
        @input=${function(e){
          inGainNode.gain.setValueAtTime(e.detail.value, audioContext.currentTime, 0.02);
        }}
        ></sc-slider>
    </div>
    <div style="padding: 4px 0;"></div>
      <sc-text> Output gain </sc-text>
      <sc-slider
        min = "0"
        max = 1
        number-box
        value=${outGainNode.gain.value}
        @input=${function(e){
          outGainNode.gain.setValueAtTime(e.detail.value, audioContext.currentTime, 0.02);
        }}
        ></sc-slider>
    </div>
    <div style="padding: 4px 0;"></div>
      <sc-text> Impulse width (ms) </sc-text>
      <sc-slider
        min = "0.1"
        max = "10"
        number-box
        value=${impulseWidth * 1000}
        @input=${function(e){
          impulseWidth = e.detail.value / 1000;
        }}
      ></sc-slider>
    </div>
  </div>
`, document.body);


