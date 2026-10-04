// 3D-Export im Browser: GLB (glTF) und OBJ – in Metern, deckungsgleich mit DXF/IFC.
import { GLTFExporter } from "./vendor/GLTFExporter.js";
import { OBJExporter } from "./vendor/OBJExporter.js";
import { buildGroup } from "./model3d.js";

export async function exportModel(model) {
  const group = buildGroup(model);
  group.updateMatrixWorld(true);
  const glb = await new GLTFExporter().parseAsync(group, { binary: true });
  const obj = "# PlanDigitalizer – 3D-Modell (Einheit: Meter, Y oben)\n" + new OBJExporter().parse(group);
  return {
    glb: new Blob([glb], { type: "model/gltf-binary" }),
    obj: new Blob([obj], { type: "model/obj" }),
  };
}
