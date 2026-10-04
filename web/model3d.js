// Gebäudemodell (aus der Erkennung) -> three.js-Gruppe.
// Modellkoordinaten: Meter, z nach oben – deckungsgleich mit DXF und IFC.
import * as THREE from "three";

const MAT = {
  wallExt: { color: 0xcbc9c2 },
  wallInt: { color: 0xe8e6e1 },
  slab: { color: 0xc7c5bf },
  stair: { color: 0xd6d3cc },
  door: { color: 0xb88c5e },
  glass: { color: 0x9fc7e6, transparent: true, opacity: 0.45 },
};

function material(key) {
  return new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0, side: THREE.DoubleSide, ...MAT[key] });
}

// Quader in Wandkoordinaten (u entlang der Wand, v quer, z hoch)
function wallBox(w, u0, u1, z0, z1, depth, mat, name) {
  const len = u1 - u0, h = z1 - z0;
  if (len < 0.005 || h < 0.005) return null;
  const a = Math.atan2(w.end[1] - w.start[1], w.end[0] - w.start[0]);
  const c = Math.cos(a), s = Math.sin(a), um = (u0 + u1) / 2;
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(len, depth, h), mat);
  mesh.position.set(w.start[0] + c * um, w.start[1] + s * um, z0 + h / 2);
  mesh.rotation.z = a;
  mesh.name = name;
  return mesh;
}

function prism(outline, z0, height, mat, name) {
  const shape = new THREE.Shape(outline.map(([x, y]) => new THREE.Vector2(x, y)));
  const geo = new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: false });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.z = z0;
  mesh.name = name;
  return mesh;
}

export function buildGroup(model, { center = false } = {}) {
  const inner = new THREE.Group();          // z-oben (wie CAD)
  const mats = Object.fromEntries(Object.keys(MAT).map((k) => [k, material(k)]));
  const z = model.elevation || 0;

  for (const w of model.walls) {
    const wall = new THREE.Group();
    wall.name = w.name;
    const m = w.external ? mats.wallExt : mats.wallInt;
    const L = Math.hypot(w.end[0] - w.start[0], w.end[1] - w.start[1]);
    const ops = [...w.openings].sort((a, b) => a.offset - b.offset);
    let cur = 0;
    const add = (o) => { if (o) wall.add(o); };
    for (const o of ops) {
      const o0 = Math.max(cur, o.offset), o1 = Math.min(L, o.offset + o.width);
      add(wallBox(w, cur, o0, z, z + w.height, w.thickness, m, w.name));
      add(wallBox(w, o0, o1, z, z + o.sill, w.thickness, m, w.name));                         // Brüstung
      add(wallBox(w, o0, o1, z + o.sill + o.height, z + w.height, w.thickness, m, w.name));   // Sturz
      const fill = o.kind === "door" ? mats.door : mats.glass;
      add(wallBox(w, o0, o1, z + o.sill, z + o.sill + o.height, o.kind === "door" ? 0.04 : 0.02, fill, o.name));
      cur = Math.max(cur, o1);
    }
    add(wallBox(w, cur, L, z, z + w.height, w.thickness, m, w.name));
    inner.add(wall);
  }
  for (const s of model.slabs) inner.add(prism(s.outline, z - s.thickness, s.thickness, mats.slab, "Bodenplatte"));
  model.stairs.forEach((st, i) => {
    const g = new THREE.Group();
    g.name = `Treppe ${String(i + 1).padStart(2, "0")}`;
    st.treads.forEach((q, j) => g.add(prism(q, z, (j + 1) * st.riser, mats.stair, g.name)));
    inner.add(g);
  });

  const root = new THREE.Group();
  root.name = "PlanDigitalizer";
  inner.rotation.x = -Math.PI / 2;           // z-oben -> y-oben (glTF/three.js)
  root.add(inner);
  if (center) {
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    const c = box.getCenter(new THREE.Vector3());
    inner.position.set(-c.x, -box.min.y, -c.z);
  }
  return root;
}
