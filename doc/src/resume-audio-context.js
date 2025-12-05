import { html, render } from 'https://unpkg.com/lit-html';

export default async function resumeAudioContext(audioContext) {
  return new Promise(resolve => {
    render(html`
      <sc-button
        style="position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; font-size: 14px;"
        @release=${async () => {
          await audioContext.resume();
          resolve();
        }}
      >Start audio and enter webpage</sc-button>
    `, document.body);
  });
}
