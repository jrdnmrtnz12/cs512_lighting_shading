function createCube() {
  const positions = new Float32Array([
    -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1,  1,  1,   // front
     1, -1, -1,  -1, -1, -1,  -1,  1, -1,   1,  1, -1,   // back
    -1,  1,  1,   1,  1,  1,   1,  1, -1,  -1,  1, -1,   // top
    -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1,  1,   // bottom
     1, -1,  1,   1, -1, -1,   1,  1, -1,   1,  1,  1,   // right
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1,  1, -1    // left
  ]);
  const faceNormals = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]];
  const normals = new Float32Array(faceNormals.flatMap(n => [...n, ...n, ...n, ...n]));

  const indices = new Uint16Array([
     0,  1,  2,   0,  2,  3,   // front
     4,  5,  6,   4,  6,  7,   // back
     8,  9, 10,   8, 10, 11,   // top
    12, 13, 14,  12, 14, 15,   // bottom
    16, 17, 18,  16, 18, 19,   // right
    20, 21, 22,  20, 22, 23    // left
  ]);
  return { positions, normals, indices };
}

function createCylinder(segments) {
  const pos = [], nrm = [], idx = [];

  // Side
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2, x = Math.cos(a), z = Math.sin(a);
    pos.push(x, -1, z, x, 1, z);
    nrm.push(x, 0, z, x, 0, z);
  }
  for (let i = 0; i < segments; i++) {
    const b = i * 2;
    idx.push(b, b + 1, b + 3, b, b + 3, b + 2);
  }

  // Caps
  for (const y of [-1, 1]) {
    const center = pos.length / 3;
    pos.push(0, y, 0);
    nrm.push(0, y, 0);
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      pos.push(Math.cos(a), y, Math.sin(a));
      nrm.push(0, y, 0);
    }
    for (let i = 1; i <= segments; i++) {
      if (y > 0) idx.push(center, center + i + 1, center + i);
      else idx.push(center, center + i, center + i + 1);
    }
  }

  return { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint16Array(idx) };
}

// #1: cone with its tip at y = 1 and a base of radius 1 at y = -1 (so h = 2, r = 1).
// the side normal leans up a bit because the side slants: (h cos a, r, h sin a) / sqrt(r^2 + h^2)
function createCone(segments) {
  const pos = [], nrm = [], idx = [];
  const h = 2, r = 1, len = Math.sqrt(r * r + h * h);

  // Side: one triangle per step, from two base points up to the tip
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2, x = Math.cos(a), z = Math.sin(a);
    const mid = a + Math.PI / segments;  // #1: the tip's normal uses the middle of the slice
    pos.push(x, -1, z, 0, 1, 0);
    nrm.push(h * x / len, r / len, h * z / len, h * Math.cos(mid) / len, r / len, h * Math.sin(mid) / len);
  }
  for (let i = 0; i < segments; i++) {
    const b = i * 2;
    idx.push(b, b + 1, b + 2);
  }

  // Base cap, facing straight down
  const center = pos.length / 3;
  pos.push(0, -1, 0);
  nrm.push(0, -1, 0);
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    pos.push(Math.cos(a), -1, Math.sin(a));
    nrm.push(0, -1, 0);
  }
  for (let i = 1; i <= segments; i++) idx.push(center, center + i, center + i + 1);

  return { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint16Array(idx) };
}

// #1: ball of radius 1, built by walking around it in a grid of latitude/longitude steps and
// placing a point at each step. on a ball of radius 1 each point already points straight out
// from the center, so the point doubles as its own normal
function createSphere(uSteps, vSteps) {
  const pos = [], nrm = [], idx = [];

  for (let i = 0; i <= vSteps; i++) {
    const v = i * Math.PI / vSteps;
    const sinv = Math.sin(v), cosv = Math.cos(v);
    for (let j = 0; j <= uSteps; j++) {
      const u = j * 2 * Math.PI / uSteps;
      const x = Math.cos(u) * sinv, y = cosv, z = Math.sin(u) * sinv;
      pos.push(x, y, z);
      nrm.push(x, y, z);
    }
  }

  // #1: connects each point to its neighbors (one step over, one row down) to make the triangles
  for (let i = 0; i < vSteps; i++) {
    for (let j = 0; j < uSteps; j++) {
      const k1 = (i * (uSteps + 1)) + j;
      const k2 = k1 + uSteps + 1;
      idx.push(k1, k2, k1 + 1);
      idx.push(k2, k2 + 1, k1 + 1);
    }
  }

  return { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint16Array(idx) };
}

// #1: donut lying flat. R is the big ring's radius and r is the tube's radius. theta goes around
// the big ring and phi goes around the tube. the normal points out from the middle of the tube:
// (cos phi cos theta, sin phi, cos phi sin theta)
function createTorus(R, r, uSteps, vSteps) {
  const pos = [], nrm = [], idx = [];

  for (let i = 0; i <= vSteps; i++) {
    const phi = i * 2 * Math.PI / vSteps;
    const sinp = Math.sin(phi), cosp = Math.cos(phi);
    for (let j = 0; j <= uSteps; j++) {
      const theta = j * 2 * Math.PI / uSteps;
      const sint = Math.sin(theta), cost = Math.cos(theta);
      pos.push((R + r * cosp) * cost, r * sinp, (R + r * cosp) * sint);
      nrm.push(cosp * cost, sinp, cosp * sint);
    }
  }

  // #1: same neighbor stitching as the ball
  for (let i = 0; i < vSteps; i++) {
    for (let j = 0; j < uSteps; j++) {
      const k1 = (i * (uSteps + 1)) + j;
      const k2 = k1 + uSteps + 1;
      idx.push(k1, k2, k1 + 1);
      idx.push(k2, k2 + 1, k1 + 1);
    }
  }

  return { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint16Array(idx) };
}

// #2: flat shading. every triangle gets its own 3 corners and one normal from the cross product of
// two of its edges, n = (p1 - p0) x (p2 - p0), so the whole triangle comes out one shade.
// triangles squeezed into a line (at a ball's poles) get skipped, and a normal that ends up
// pointing into the shape gets flipped so it points out like the smooth normals do
function makeFlat({ positions, normals, indices }) {
  const pos = [], nrm = [], idx = [];
  const corner = (list, k) => [list[k * 3], list[k * 3 + 1], list[k * 3 + 2]];

  for (let t = 0; t < indices.length; t += 3) {
    const p0 = corner(positions, indices[t]);
    const p1 = corner(positions, indices[t + 1]);
    const p2 = corner(positions, indices[t + 2]);
    let n = cross(subtract(p1, p0), subtract(p2, p0));
    const length = Math.sqrt(dot(n, n));
    if (length < 1e-8) continue;
    n = n.map(c => c / length);

    const smooth = [0, 1, 2].map(c =>
      normals[indices[t] * 3 + c] + normals[indices[t + 1] * 3 + c] + normals[indices[t + 2] * 3 + c]);
    if (dot(n, smooth) < 0) n = n.map(c => -c);

    const k = pos.length / 3;
    pos.push(...p0, ...p1, ...p2);
    nrm.push(...n, ...n, ...n);
    idx.push(k, k + 1, k + 2);
  }

  return { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint16Array(idx) };
}
