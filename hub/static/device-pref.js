/* MAVIS camera/microphone rule, loaded first on every MAVIS page (the modules' own code is unchanged):
   every page uses the laptop's own camera and microphone - never the virtual "DroidCam" devices that the DroidCam
   Windows client adds, when another device exists. (The OCR + LLM module streams from the DroidCam phone app over
   Wi-Fi through its own "Connect and start scanning" form, which needs no Windows client.)
   Applies only when a page asks without naming a device; a page that names a device (the OCR page's
   "Use the laptop camera" button) gets exactly that device. If the chosen device fails, the browser default is used. */
(function () {
  'use strict';
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia || md.__mavisPref) return;
  md.__mavisPref = true;
  const orig = md.getUserMedia.bind(md);
  const RE = /droidcam/i;
  const PREFER_PHONE = false;
  window.MAVIS_DEVICE_RULE = PREFER_PHONE ? 'phone (DroidCam)' : 'laptop';

  async function pick(kind) {
    let list = [];
    try { list = (await md.enumerateDevices()).filter((d) => d.kind === kind); } catch (e) { return null; }
    if (!list.some((d) => RE.test(d.label))) return null;          // no DroidCam device (or names hidden): browser default
    const d = PREFER_PHONE ? list.find((x) => RE.test(x.label)) : list.find((x) => x.label && !RE.test(x.label));
    return d ? d.deviceId : null;
  }
  const named = (v) => v && typeof v === 'object' && v.deviceId;
  const asObj = (v) => (v === true ? {} : { ...v });
  async function choose(c) {
    const out = { ...c };
    if (out.video && !named(out.video)) { const id = await pick('videoinput'); if (id) out.video = { ...asObj(out.video), deviceId: { exact: id } }; }
    if (out.audio && !named(out.audio)) { const id = await pick('audioinput'); if (id) out.audio = { ...asObj(out.audio), deviceId: { exact: id } }; }
    return out;
  }
  window.MAVIS_findCamera = async function (phone) {            // used by the OCR page's two buttons
    try {
      const cams = (await md.enumerateDevices()).filter((d) => d.kind === 'videoinput');
      const d = phone ? cams.find((x) => RE.test(x.label)) : cams.find((x) => x.label && !RE.test(x.label));
      return d ? d.deviceId : null;
    } catch (e) { return null; }
  };
  md.getUserMedia = async function (constraints) {
    const c = constraints || {};
    let want = await choose(c);
    try {
      const s = await orig(want);
      // before the first permission the device names are hidden: decide again now that they are visible
      const again = await choose(c);
      if (JSON.stringify(again) !== JSON.stringify(want)) {
        s.getTracks().forEach((t) => t.stop());
        want = again;
        return await orig(want);
      }
      return s;
    } catch (e) {
      if (JSON.stringify(want) !== JSON.stringify(c)) return orig(c);
      throw e;
    }
  };
})();
