const VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
in vec3 aNormal;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;   // (M^-1)^T, keeps normals straight when a shape is stretched

out vec3 vPosition;
out vec3 vNormal;

void main() {
  vec4 p = uModelView * vec4(aPosition, 1.0);
  vPosition = p.xyz;
  vNormal = normalize(uNormalMatrix * aNormal);
  gl_Position = uProjection * p;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec3 vPosition;
in vec3 vNormal;

uniform vec3 uColor;
uniform float uKa, uKd, uKs, uAlpha;   // material: ambient, diffuse, specular, shininess
uniform float uAmbient;                // how much ambient light reaches this object
uniform bool uBlinn;                   // true = Blinn-Phong highlight, false = Phong

uniform vec3 uLightPos;   // eye space (room light)
uniform vec3 uLightColor;

uniform vec3 uSpotPos;    // eye space (lamp light)
uniform vec3 uSpotDir;
uniform vec3 uSpotColor;
uniform float uSpotCutoff;   // cos of the cone's half angle
uniform float uSpotExp;      // how soft the edge of the cone is

out vec4 fragColor;

// diffuse + specular from one light
vec3 calculateLight(vec3 lightPos, vec3 lightColor, vec3 N, vec3 V) {
  vec3 L = normalize(lightPos - vPosition);
  float diff = max(dot(N, L), 0.0);

  float spec;
  if (uBlinn) {
    vec3 H = normalize(L + V);               // halfway vector
    spec = pow(max(dot(N, H), 0.0), uAlpha);
  } else {
    vec3 R = reflect(-L, N);
    spec = pow(max(dot(V, R), 0.0), uAlpha);
  }
  if (diff == 0.0) spec = 0.0;               // no highlight on the side facing away from the light

  return lightColor * (uKd * diff * uColor + uKs * spec * vec3(1.0));
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(-vPosition);           // camera sits at the origin

  // ambient is only added once, not per light
  vec3 ambient = uAmbient * uKa * uColor;

  // light 1: the room light
  vec3 room = calculateLight(uLightPos, uLightColor, N, V);

  // light 2: the lamp's spotlight, only inside its cone, fading with distance
  float spotCos = dot(normalize(vPosition - uSpotPos), normalize(uSpotDir));
  float spot = spotCos > uSpotCutoff ? pow(spotCos, uSpotExp) : 0.0;
  float d = length(uSpotPos - vPosition);
  float atten = 1.0 / (1.0 + 0.09 * d + 0.032 * d * d);
  vec3 lamp = spot * atten * calculateLight(uSpotPos, uSpotColor, N, V);

  fragColor = vec4(ambient + room + lamp, 1.0);
}
`;

// WebGL setup
const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl2");

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

// Fixed attribute slots, so the meshes stay valid when the shaders are recompiled
const ATTRIBUTES = { aPosition: 0, aNormal: 1 };

function createProgram(vsSource, fsSource) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource));
  for (const name in ATTRIBUTES) gl.bindAttribLocation(prog, ATTRIBUTES[name], name);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// Shader editors
const vertEditor = document.getElementById("vertEditor");
const fragEditor = document.getElementById("fragEditor");
const shaderError = document.getElementById("shaderError");
vertEditor.value = VERTEX_SHADER;
fragEditor.value = FRAGMENT_SHADER;

let program, uModelView, uProjection, uLightPos, uColor;

// #2: the new uniforms for materials, shading and the second light, looked up by name
const EXTRA_UNIFORMS = ["uNormalMatrix", "uKa", "uKd", "uKs", "uAlpha", "uAmbient", "uBlinn", "uLightColor",
  "uSpotPos", "uSpotDir", "uSpotColor", "uSpotCutoff", "uSpotExp"];
const u = {};

function updateProgram() {
  try {
    const prog = createProgram(vertEditor.value, fragEditor.value);
    gl.deleteProgram(program);
    program = prog;
    gl.useProgram(program);
    uModelView = gl.getUniformLocation(program, "uModelView");
    uProjection = gl.getUniformLocation(program, "uProjection");
    uLightPos = gl.getUniformLocation(program, "uLightPos");
    uColor = gl.getUniformLocation(program, "uColor");
    for (const name of EXTRA_UNIFORMS) u[name] = gl.getUniformLocation(program, name);
    shaderError.textContent = "";
  } catch (e) {
    shaderError.textContent = e.message.replace(/\0/g, "");  // some drivers end the log with \0
  }
}
updateProgram();
vertEditor.addEventListener("input", updateProgram);
fragEditor.addEventListener("input", updateProgram);

function bindAttribute(name, data) {
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.vertexAttribPointer(ATTRIBUTES[name], 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(ATTRIBUTES[name]);
}

function createMesh({ positions, normals, indices }) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  bindAttribute("aPosition", positions);
  bindAttribute("aNormal", normals);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  return { vao, count: indices.length };
}

const cubeMesh = createMesh(createCube());
const cylinderMesh = createMesh(createCylinder(32));

// #1: the three new primitives: cone, sphere, torus (plus the cube and cylinder above = 5 kinds)
const coneMesh = createMesh(createCone(32));
const sphereMesh = createMesh(createSphere(32, 16));
const torusMesh = createMesh(createTorus(1, 0.12, 32, 12));

// #2: flat-shaded copies of the curved shapes, one normal per triangle
const flatCylinderMesh = createMesh(makeFlat(createCylinder(16)));
const flatConeMesh = createMesh(makeFlat(createCone(16)));
const flatSphereMesh = createMesh(makeFlat(createSphere(16, 10)));
const flatTorusMesh = createMesh(makeFlat(createTorus(1, 0.12, 16, 8)));

gl.enable(gl.DEPTH_TEST);
gl.clearColor(0, 0, 0, 1);
const projection = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
const lightWorld = [3, 4, 5];

// nteractive controls
let dragRotX = 0, dragRotY = -0.5;      // mouse drag
let camX = 0, camY = 0, camZ = -7;      // arrow keys, w, s
let sliderAngle = 0;                    // degrees
let spinning = false, spinAngle = 0;    // Start/Stop button

let mouseDown = false, lastX, lastY;
canvas.addEventListener("mousedown", e => { mouseDown = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => mouseDown = false);
window.addEventListener("mousemove", e => {
  if (!mouseDown) return;
  dragRotY += (e.clientX - lastX) * 0.01;
  dragRotX += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});

document.addEventListener("keydown", e => {
  if (e.target === vertEditor || e.target === fragEditor) return;  // let the editors use the keys
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;  // #4: and the new sliders/dropdowns
  const step = 0.2;
  switch (e.key) {
    case "ArrowUp": camY -= step; break;
    case "ArrowDown": camY += step; break;
    case "ArrowLeft": camX += step; break;
    case "ArrowRight": camX -= step; break;
    case "w": camZ += step; break;
    case "s": camZ -= step; break;
    default: return;
  }
  e.preventDefault();  // keep the arrow keys from scrolling the page
});

const axisSelect = document.getElementById("axisSelect");
document.getElementById("slider").addEventListener("input", e => {
  sliderAngle = Number(e.target.value);
  document.getElementById("sliderValue").textContent = sliderAngle;
});
document.getElementById("spinButton").addEventListener("click", () => spinning = !spinning);


// #2: material presets. ka, kd, ks are how much ambient, diffuse and specular light the surface
// gives back, and alpha is the shininess (small = wide soft highlight, big = tiny sharp one)
const MATERIALS = {
  matte:   { ka: 0.2, kd: 0.8, ks: 0.0, alpha: 1 },     // diffuse only, no highlight at all
  plastic: { ka: 0.2, kd: 0.8, ks: 0.5, alpha: 8 },
  metal:   { ka: 0.2, kd: 0.5, ks: 0.9, alpha: 120 },
  gold:    { ka: 0.25, kd: 0.8, ks: 1.0, alpha: 100, color: [0.9, 0.65, 0.1] },
  glow:    { ka: 1.0, kd: 0.0, ks: 0.0, alpha: 1 }      // the bulb: ambient only so it looks lit up
};

function createNode(mesh, color, shape, material = MATERIALS.plastic) {
  return { mesh, color, shape, material, joint: null, children: [] };
}

// #1: level 0, the root. the desk is a flat stretched cube with its top at y = 0
const desk = createNode(cubeMesh, [0.55, 0.38, 0.22],
  multiply(translate(0, -0.1, 0), scale(2.6, 0.1, 1.6)), MATERIALS.matte);

// #1: level 1. the lamp's base, a short wide cylinder sitting on the desk
const lampBase = createNode(cylinderMesh, [0.3, 0.3, 0.35],
  multiply(translate(0, 0.1, 0), scale(0.45, 0.1, 0.45)));

// Lower arm: scale the cylinder then move it up by 0.6 so its bottom sits at the joint
const lowerArm = createNode(cylinderMesh, [1, 0.2, 0.2],
  multiply(translate(0, 0.6, 0), scale(0.3, 0.6, 0.3)));

// Upper arm: scale the cube then move it up by 0.5 so its bottom sits at the joint
const upperArm = createNode(cubeMesh, [0.2, 0.4, 1],
  multiply(translate(0, 0.5, 0), scale(0.2, 0.5, 0.2)));

// #1: level 4. the shade, a cone with its tip at the end of the upper arm and its wide side facing down
const shade = createNode(coneMesh, [0.15, 0.55, 0.35],
  multiply(translate(0, -0.3, 0), scale(0.35, 0.3, 0.35)));

// #1: level 5. the bulb, a small ball tucked under the shade. it glows, so it uses the lamp's light
// for its ambient instead of the room's
const bulb = createNode(sphereMesh, [1, 0.92, 0.7],
  multiply(translate(0, -0.6, 0), scale(0.15, 0.15, 0.15)), MATERIALS.glow);
bulb.glows = true;

// #1: level 5. a donut ring around the bottom edge of the shade
const rim = createNode(torusMesh, [0.85, 0.75, 0.3],
  multiply(translate(0, -0.6, 0), scale(0.35, 0.35, 0.35)));

// #2: level 1. two copies of the same ball on the desk, one smooth and one flat
const smoothBall = createNode(sphereMesh, [0.25, 0.5, 1],
  multiply(translate(0, 0.35, 0), scale(0.35, 0.35, 0.35)));
const flatBall = createNode(flatSphereMesh, [1, 0.45, 0.1],
  multiply(translate(0, 0.35, 0), scale(0.35, 0.35, 0.35)));

// #1: the tree: desk -> lamp base -> lower arm -> upper arm -> shade -> bulb and rim,
// and the two balls hang off the desk
desk.children.push(lampBase, smoothBall, flatBall);
lampBase.children.push(lowerArm);
lowerArm.children.push(upperArm);
upperArm.children.push(shade);
shade.children.push(bulb, rim);

// #1: parts that never move just sit at a fixed spot in their parent
bulb.joint = translate(0, 0, 0);
rim.joint = translate(0, 0, 0);
smoothBall.joint = translate(0.3, 0, 0.55);
flatBall.joint = translate(1.4, 0, -0.35);

// #2: flat versions of the lamp parts, used when Lamp shading is set to Flat
// (the cube's faces are already flat, so the upper arm keeps its mesh)
lampBase.flatMesh = flatCylinderMesh;
lowerArm.flatMesh = flatCylinderMesh;
upperArm.flatMesh = cubeMesh;
shade.flatMesh = flatConeMesh;
rim.flatMesh = flatTorusMesh;

// #4: lamp joint angles in degrees from the new sliders, and the light settings
const DEG = Math.PI / 180;
const DEFAULT_JOINTS = { swivel: 0, lower: 20, upper: -15, shade: 40 };   // #4: stays above the desk even at the ends of the sway
const DEFAULT_LIGHTS = { spotBright: 150, spotWidth: 30, roomBright: 50 };
let joint = Object.assign({}, DEFAULT_JOINTS);
let lights = Object.assign({}, DEFAULT_LIGHTS);
let swayOn = true, spotOn = true, lampFlat = false, blinn = false;
let spotColor = [1, 0.95, 0.8];
let spotLevel = 1, roomLevel = 1;   // worked out from the settings every frame

// #4: each row is the setting, its slider, the text showing its value, and the unit
const jointControls = [
  ['swivel', 'swivelSlider', 'swivelOut', '°'],
  ['lower', 'lowerSlider', 'lowerOut', '°'],
  ['upper', 'upperSlider', 'upperOut', '°'],
  ['shade', 'shadeSlider', 'shadeOut', '°']
];
const lightControls = [
  ['spotBright', 'spotBrightSlider', 'spotBrightOut', '%'],
  ['spotWidth', 'spotWidthSlider', 'spotWidthOut', '°'],
  ['roomBright', 'roomBrightSlider', 'roomBrightOut', '%']
];

// #4: hooks up one group of sliders: dragging saves the value and updates the number next to it,
// and letting go un-focuses the slider so the camera keys work again
function wireSliders(rows, values) {
  rows.forEach(([key, sliderId, outId, unit]) => {
    const slider = document.getElementById(sliderId);
    slider.addEventListener("input", e => {
      values()[key] = Number(e.target.value);
      document.getElementById(outId).textContent = values()[key] + unit;
    });
    slider.addEventListener("change", () => slider.blur());
  });
}
wireSliders(jointControls, () => joint);
wireSliders(lightControls, () => lights);

document.getElementById("swayCheck").addEventListener("change", e => swayOn = e.target.checked);
document.getElementById("spotCheck").addEventListener("change", e => spotOn = e.target.checked);

// #4: the color picker gives "#rrggbb", which gets turned into 0-1 numbers for the shader
function hexToRgb(hex) {
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
}
document.getElementById("spotColor").addEventListener("input", e => spotColor = hexToRgb(e.target.value));

// #4: each material dropdown, its starting preset, and the parts it changes
const materialControls = [
  ['armsMaterial', 'metal', [lampBase, lowerArm, upperArm]],
  ['shadeMaterial', 'plastic', [shade, rim]],
  ['smoothBallMaterial', 'metal', [smoothBall]],
  ['flatBallMaterial', 'matte', [flatBall]]
];
function applyMaterial(selectId, nodes) {
  const material = MATERIALS[document.getElementById(selectId).value];
  nodes.forEach(n => n.material = material);
}
materialControls.forEach(([selectId, start, nodes]) => {
  document.getElementById(selectId).value = start;
  applyMaterial(selectId, nodes);
  document.getElementById(selectId).addEventListener("change", () => applyMaterial(selectId, nodes));
});

document.getElementById("lampShading").addEventListener("change", e => lampFlat = e.target.value === "flat");
document.getElementById("specModel").addEventListener("change", e => blinn = e.target.value === "blinn");

// #4: puts the new lamp, light and material controls back to how they started
function resetAll() {
  joint = Object.assign({}, DEFAULT_JOINTS);
  lights = Object.assign({}, DEFAULT_LIGHTS);
  jointControls.forEach(([key, sliderId, outId, unit]) => {
    document.getElementById(sliderId).value = joint[key];
    document.getElementById(outId).textContent = joint[key] + unit;
  });
  lightControls.forEach(([key, sliderId, outId, unit]) => {
    document.getElementById(sliderId).value = lights[key];
    document.getElementById(outId).textContent = lights[key] + unit;
  });
  swayOn = true; spotOn = true; lampFlat = false; blinn = false;
  document.getElementById("swayCheck").checked = true;
  document.getElementById("spotCheck").checked = true;
  document.getElementById("lampShading").value = "smooth";
  document.getElementById("specModel").value = "phong";
  document.getElementById("spotColor").value = "#fff2cc";
  spotColor = hexToRgb("#fff2cc");
  materialControls.forEach(([selectId, start, nodes]) => {
    document.getElementById(selectId).value = start;
    applyMaterial(selectId, nodes);
  });
}
document.getElementById("resetBtn").addEventListener("click", resetAll);

function drawNode(node, frame) {
  const modelView = multiply(frame, node.shape);
  gl.uniformMatrix4fv(uModelView, false, modelView);
  // #2: the normal matrix for this part, so stretched shapes still get normals that stick straight out
  gl.uniformMatrix3fv(u.uNormalMatrix, false, normalMatrix(modelView));

  // #2: this part's material. a preset with its own color (gold) overrides the part's color
  const m = node.material;
  gl.uniform3fv(uColor, m.color || node.color);
  gl.uniform1f(u.uKa, m.ka);
  gl.uniform1f(u.uKd, m.kd);
  gl.uniform1f(u.uKs, m.ks);
  gl.uniform1f(u.uAlpha, m.alpha);
  gl.uniform1f(u.uAmbient, node.glows ? Math.max(spotLevel, 0.15) : roomLevel);

  // #2: lamp parts switch to their flat mesh when Lamp shading is set to Flat
  const mesh = (lampFlat && node.flatMesh) ? node.flatMesh : node.mesh;
  gl.bindVertexArray(mesh.vao);
  gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
}

//Data structure
function drawTree(root, rootFrame) {
  const stack = [[root, rootFrame]];
  while (stack.length > 0) {
    const [node, parentFrame] = stack.pop();
    const frame = multiply(parentFrame, node.joint);
    drawNode(node, frame);
    for (const child of node.children) stack.push([child, frame]);
  }
}

// Render loop
let lastTime = 0;
function render(time) {
  const t = time / 1000;
  if (spinning) spinAngle += t - lastTime;
  lastTime = t;
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Camera moves with the keys; the light stays fixed in the world.
  const view = translate(camX, camY, camZ);
  gl.uniformMatrix4fv(uProjection, false, projection);
  gl.uniform3fv(uLightPos, transformPoint(view, lightWorld));

  // Whole-model rotation
  const axisRotation = ROTATE[axisSelect.value](sliderAngle * Math.PI / 180 + spinAngle);
  const rotation = multiply(multiply(rotateY(dragRotY), rotateX(dragRotX)), axisRotation);

  // #1: the desk is the root now, so the whole-model rotation goes on it
  desk.joint = multiply(translate(0, -2, 0), rotation);

  // #4: lamp joints from the sliders. with Sway on, the original back-and-forth motion gets added on top
  const sway = swayOn ? 1 : 0;
  lampBase.joint = multiply(translate(-1.2, 0, 0), rotateY(joint.swivel * DEG));

  // Lower arm
  lowerArm.joint = multiply(translate(0, 0.2, 0), rotateZ(joint.lower * DEG + sway * 0.5 * Math.sin(t)));

  // Upper arm (child)
  upperArm.joint = multiply(translate(0, 1.2, 0), rotateZ(joint.upper * DEG + sway * Math.sin(1.5 * t)));

  shade.joint = multiply(translate(0, 1.0, 0), rotateZ(joint.shade * DEG));

  // #3: light brightness from the sliders and checkbox
  spotLevel = spotOn ? lights.spotBright / 100 : 0;
  roomLevel = lights.roomBright / 100;
  gl.uniform3fv(u.uLightColor, [roomLevel, roomLevel, roomLevel]);
  gl.uniform3fv(u.uSpotColor, spotColor.map(c => c * spotLevel));
  gl.uniform1f(u.uSpotCutoff, Math.cos(lights.spotWidth * DEG));
  gl.uniform1f(u.uSpotExp, 10);
  gl.uniform1i(u.uBlinn, blinn ? 1 : 0);

  // #5: the spotlight rides on the lamp. multiplying the same chain of joints the tree uses gives the
  // shade's matrix, then the bulb's spot is a point (w = 1) and "straight down out of the shade" is a
  // direction (w = 0), so moving any joint aims the light
  let shadeFrame = view;
  for (const n of [desk, lampBase, lowerArm, upperArm, shade]) shadeFrame = multiply(shadeFrame, n.joint);
  gl.uniform3fv(u.uSpotPos, transformPoint(shadeFrame, [0, -0.6, 0]));
  gl.uniform3fv(u.uSpotDir, transformDirection(shadeFrame, [0, -1, 0]));

  drawTree(desk, view);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
