// 3D-Vorschau (three.js, lokal eingebunden – keine externen Server)
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { buildGroup } from "./model3d.js";

export function mount(container, model) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 2000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.maxPolarAngle = Math.PI * 0.495;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b4ac, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(-0.6, 1.4, 0.9);
  scene.add(sun);

  const group = buildGroup(model, { center: true });
  group.traverse((o) => {
    if (o.isMesh && !o.material.transparent) {
      o.add(new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 25),
        new THREE.LineBasicMaterial({ color: 0x5b5f66, transparent: true, opacity: 0.5 })));
    }
  });
  scene.add(group);
  group.updateMatrixWorld(true);
  const size = new THREE.Box3().setFromObject(group).getSize(new THREE.Vector3());
  const grid = new THREE.GridHelper(Math.ceil(Math.max(size.x, size.z) * 1.6), Math.ceil(Math.max(size.x, size.z) * 1.6), 0xc9ccd1, 0xe1e3e6);
  grid.position.y = -0.002;
  scene.add(grid);

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  function fit() {
    const r = Math.max(size.x, size.y, size.z) || 10;
    controls.target.set(0, size.y * 0.3, 0);
    camera.position.set(-r * 0.75, r * 0.85, r * 1.05);
    camera.near = r / 200;
    camera.far = r * 50;
    camera.updateProjectionMatrix();
    controls.update();
  }

  let raf = 0;
  const loop = () => { raf = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();
  fit();
  loop();

  return {
    fit,
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); });
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
