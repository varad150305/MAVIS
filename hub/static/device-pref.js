/* MAVIS: use the phone (DroidCam) for camera and microphone on every page.
   Loaded first on every MAVIS page (launcher, gesture control, translation, OCR, Drishti). When a page asks for the
   camera or microphone without naming a device, the DroidCam device is used if one exists ("DroidCam Source" camera,
   "DroidCam Virtual Audio" microphone); otherwise the browser's normal choice. The modules' own code is unchanged. */
(function () {
  'use strict';
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia || md.__mavisPref) return;
  md.__mavisPref = true;
  const orig = md.getUserMedia.bind(md);
  const RE = /droidcam/i;
  async function find(kind) {
    try { const d = (await md.enumerateDevices()).find((x) => x.kind === kind && RE.test(x.label)); return d ? d.deviceId : null; }
    catch (e) { return null; }
  }
  const asObj = (v) => (v === true ? {} : { ...v });
  async function prefer(c) {
    const out = { ...c };
    if (out.video && !(typeof out.video === 'object' && out.video.deviceId)) { const id = await find('videoinput'); if (id) out.video = { ...asObj(out.video), deviceId: { exact: id } }; }
    if (out.audio && !(typeof out.audio === 'object' && out.audio.deviceId)) { const id = await find('audioinput'); if (id) out.audio = { ...asObj(out.audio), deviceId: { exact: id } }; }
    return out;
  }
  md.getUserMedia = async function (constraints) {
    const c = constraints || {};
    let want = await prefer(c);
    try {
      const s = await orig(want);
      // Before the first permission the device names are hidden; look again now that they are visible.
      const again = await prefer(c);
      if (JSON.stringify(again) !== JSON.stringify(want)) {
        s.getTracks().forEach((t) => t.stop());
        want = again;
        return await orig(want);
      }
      return s;
    } catch (e) {
      if (want !== c) return orig(c);   // the DroidCam device refused the request: fall back to the default
      throw e;
    }
  };
})();
