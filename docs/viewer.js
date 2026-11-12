"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("scene");
const status = $("status");
const subtract = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalized = (a) => {
  const length = Math.hypot(...a) || 1;
  return a.map((v) => v / length);
};
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function multiply(a, b) {
  const result = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      for (let k = 0; k < 4; k++)
        result[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
    }
  }
  return result;
}

function perspective(aspect) {
  const f = 1 / Math.tan(Math.PI / 7);
  const near = 0.01;
  const far = 150;
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) / (near - far),
    -1,
    0,
    0,
    (2 * far * near) / (near - far),
    0,
  ]);
}

function lookAt(eye, target) {
  const z = normalized(subtract(eye, target));
  const x = normalized(cross([0, -1, 0], z));
  const y = cross(z, x);
  return new Float32Array([
    x[0],
    y[0],
    z[0],
    0,
    x[1],
    y[1],
    z[1],
    0,
    x[2],
    y[2],
    z[2],
    0,
    -dot(x, eye),
    -dot(y, eye),
    -dot(z, eye),
    1,
  ]);
}

function createProgram(gl) {
  const vertexSource = `
    attribute vec3 position;
    attribute vec3 color;
    uniform mat4 transform;
    uniform float size;
    varying vec3 pointColor;
    void main() {
      gl_Position = transform * vec4(position, 1.0);
      gl_PointSize = size;
      pointColor = color;
    }`;
  const fragmentSource = `
    precision mediump float;
    varying vec3 pointColor;
    uniform bool isPoint;
    void main() {
      if (isPoint && distance(gl_PointCoord, vec2(0.5)) > 0.5) discard;
      gl_FragColor = vec4(pointColor, 1.0);
    }`;
  const program = gl.createProgram();
  for (const [kind, source] of [
    [gl.VERTEX_SHADER, vertexSource],
    [gl.FRAGMENT_SHADER, fragmentSource],
  ]) {
    const shader = gl.createShader(kind);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(program));
  return program;
}

function setupViewer(scene) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: true });
  if (!gl)
    throw new Error(
      "WebGL is unavailable. You can still download the reconstruction below.",
    );
  const program = createProgram(gl);
  gl.useProgram(program);
  gl.enable(gl.DEPTH_TEST);
  const positionLocation = gl.getAttribLocation(program, "position");
  const colorLocation = gl.getAttribLocation(program, "color");
  const transformLocation = gl.getUniformLocation(program, "transform");
  const sizeLocation = gl.getUniformLocation(program, "size");
  const pointLocation = gl.getUniformLocation(program, "isPoint");
  const bounds = [0, 1, 2].map((axis) => {
    const values = scene.points.map((p) => p[axis]).sort((a, b) => a - b);
    return [
      values[Math.floor(values.length * 0.02)],
      values[Math.floor(values.length * 0.98)],
    ];
  });
  const center = scene.view_target || bounds.map(([a, b]) => (a + b) / 2);
  const radius = Math.max(...bounds.map(([a, b]) => b - a)) / 2 || 1;
  const transformPoint = (p) => p.map((v, i) => (v - center[i]) / radius);
  function buffer(points, colors) {
    const positions = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positions);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array(points.flat()),
      gl.STATIC_DRAW,
    );
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array(colors.flat()),
      gl.STATIC_DRAW,
    );
    return { positions, colors: colorBuffer, count: points.length };
  }
  const finalCloud = buffer(
    scene.points.map(transformPoint),
    scene.colors.map((c) => c.map((v) => v / 255)),
  );
  const initialCloud = buffer(
    scene.initial_points.map(transformPoint),
    scene.initial_points.map(() => [0.85, 0.79, 0.64]),
  );
  let cameraBuffer;
  let currentFrame = 0;
  function updateCameras() {
    const positions = [];
    const colors = [];
    const initial = $("stage").value === "initial";
    const indices = initial
      ? scene.initialization.pair
      : scene.poses.map((_, i) => i);
    for (const index of indices) {
      const pose = initial
        ? scene.initialization.poses[indices.indexOf(index)]
        : scene.poses[index];
      const c = [0, 1, 2].map(
        (axis) =>
          -pose.slice(0, 3).reduce((s, row) => s + row[axis] * row[3], 0),
      );
      const scale = radius * 0.022;
      const local = [
        [-1, -0.75, 1.5],
        [1, -0.75, 1.5],
        [1, 0.75, 1.5],
        [-1, 0.75, 1.5],
      ];
      const corners = local.map((p) =>
        c.map(
          (v, axis) =>
            v +
            scale *
              pose.slice(0, 3).reduce((s, row, k) => s + row[axis] * p[k], 0),
        ),
      );
      const color =
        index === currentFrame ? [1.0, 0.57, 0.28] : [0.57, 0.72, 0.64];
      for (let j = 0; j < 4; j++) {
        positions.push(
          transformPoint(c),
          transformPoint(corners[j]),
          transformPoint(corners[j]),
          transformPoint(corners[(j + 1) % 4]),
        );
        colors.push(color, color, color, color);
      }
    }
    if (cameraBuffer) {
      gl.deleteBuffer(cameraBuffer.positions);
      gl.deleteBuffer(cameraBuffer.colors);
    }
    cameraBuffer = buffer(positions, colors);
  }
  updateCameras();
  let yaw;
  let pitch;
  let distance;
  let target;
  let scheduled = false;
  function schedule() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(render);
    }
  }
  function reset() {
    const offset = subtract(scene.view_position || [0, 0, 0], center);
    const direction = normalized(offset);
    yaw = Math.atan2(direction[0], direction[2]);
    pitch = clamp(Math.asin(direction[1]), -1.4, 1.4);
    distance = Math.max(0.2, Math.hypot(...offset) / radius);
    target = [0, 0, 0];
    schedule();
  }
  function draw(data, mode) {
    gl.bindBuffer(gl.ARRAY_BUFFER, data.positions);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, data.colors);
    gl.enableVertexAttribArray(colorLocation);
    gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
    gl.uniform1i(pointLocation, mode === gl.POINTS ? 1 : 0);
    gl.drawArrays(mode, 0, data.count);
  }
  function render() {
    scheduled = false;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth * dpr);
    const height = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const eye = target.map(
      (v, i) =>
        v +
        distance *
          [
            Math.sin(yaw) * Math.cos(pitch),
            Math.sin(pitch),
            Math.cos(yaw) * Math.cos(pitch),
          ][i],
    );
    gl.uniformMatrix4fv(
      transformLocation,
      false,
      multiply(perspective(width / height), lookAt(eye, target)),
    );
    gl.uniform1f(sizeLocation, Number($("point-size").value) * dpr);
    const cloud = $("stage").value === "initial" ? initialCloud : finalCloud;
    draw(cloud, gl.POINTS);
    if ($("cameras").checked) draw(cameraBuffer, gl.LINES);
    $("render-count").textContent =
      `${cloud.count.toLocaleString()} displayed points`;
  }
  let pointers = new Map();
  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  });
  canvas.addEventListener("pointermove", (event) => {
    const previous = pointers.get(event.pointerId);
    if (!previous) return;
    const dx = event.clientX - previous.x;
    const dy = event.clientY - previous.y;
    if (pointers.size > 1) {
      const other = [...pointers.entries()].find(
        ([id]) => id !== event.pointerId,
      )[1];
      const oldDistance = Math.hypot(
        previous.x - other.x,
        previous.y - other.y,
      );
      const newDistance = Math.hypot(
        event.clientX - other.x,
        event.clientY - other.y,
      );
      if (newDistance > 1)
        distance = clamp((distance * oldDistance) / newDistance, 0.1, 60);
    } else if (event.shiftKey || event.buttons === 2) {
      const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
      target = target.map((v, i) => v + right[i] * dx * distance * 0.0015);
      target[1] -= dy * distance * 0.0015;
    } else {
      yaw -= dx * 0.007;
      pitch = clamp(pitch - dy * 0.007, -1.5, 1.5);
    }
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    schedule();
  });
  const release = (event) => pointers.delete(event.pointerId);
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("lostpointercapture", release);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      distance = clamp(distance * Math.exp(event.deltaY * 0.001), 0.1, 60);
      schedule();
    },
    { passive: false },
  );
  canvas.addEventListener("keydown", (event) => {
    const actions = {
      ArrowLeft: () => {
        yaw += 0.1;
      },
      ArrowRight: () => {
        yaw -= 0.1;
      },
      ArrowUp: () => {
        pitch = clamp(pitch + 0.1, -1.5, 1.5);
      },
      ArrowDown: () => {
        pitch = clamp(pitch - 0.1, -1.5, 1.5);
      },
      "+": () => {
        distance = Math.max(0.1, distance * 0.9);
      },
      "=": () => {
        distance = Math.max(0.1, distance * 0.9);
      },
      "-": () => {
        distance = Math.min(60, distance * 1.1);
      },
      Home: reset,
    };
    if (actions[event.key]) {
      event.preventDefault();
      actions[event.key]();
      schedule();
    }
  });
  $("reset").addEventListener("click", reset);
  $("stage").addEventListener("change", () => {
    $("stage-label").textContent = $("stage").selectedOptions[0].textContent;
    updateCameras();
    schedule();
  });
  $("cameras").addEventListener("change", schedule);
  $("point-size").addEventListener("input", () => {
    $("size-value").value = Number($("point-size").value).toFixed(1);
    schedule();
  });
  $("frame-slider").max = scene.image_names.length - 1;
  function selectFrame() {
    currentFrame = Number($("frame-slider").value);
    $("frame").src =
      `assets/frames/${String(currentFrame).padStart(3, "0")}.jpg`;
    $("frame").alt =
      `Input photograph ${currentFrame + 1} of ${scene.image_names.length}, ${scene.image_names[currentFrame]}`;
    $("frame-count").textContent =
      `${String(currentFrame + 1).padStart(2, "0")} / ${scene.image_names.length}`;
    $("frame-name").textContent = scene.image_names[currentFrame];
    updateCameras();
    schedule();
  }
  $("frame-slider").addEventListener("input", selectFrame);
  new ResizeObserver(schedule).observe(canvas);
  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    status.textContent =
      "The 3D graphics context was interrupted. Reload this page to restore the viewer.";
    status.hidden = false;
  });
  reset();
  selectFrame();
  status.hidden = true;
}

function addResult(name, cameras, points, stats, detail = {}) {
  const row = document.createElement("tr");
  const trackViews = detail.mean_track_length?.toFixed(2) || "—";
  const agreement = detail.pairwise_consistency
    ? `${(detail.pairwise_consistency.fraction * 100).toFixed(1)}%`
    : "—";
  const ba = detail.bundle_adjustment;
  const baStatus = !ba?.enabled
    ? "Disabled"
    : ba.converged
      ? "Converged"
      : `Budget ${ba.evaluations}`;

  for (const value of [
    name,
    cameras,
    points.toLocaleString(),
    `${stats.mean_px.toFixed(3)} px`,
    `${stats.median_px.toFixed(3)} px`,
    trackViews,
    agreement,
    baStatus,
  ]) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
  }
  $("results-body").appendChild(row);
}

async function load() {
  try {
    const response = await fetch("assets/scene.json");
    if (!response.ok)
      throw new Error(`Scene download failed (${response.status}).`);
    const scene = await response.json();
    if (!scene.points?.length)
      throw new Error("The reconstruction contains no points.");
    const metrics = scene.metrics;
    const poseLabel =
      metrics.configuration.pose_method === "essential"
        ? "Essential-only"
        : "PnP";
    $("camera-count").textContent =
      `${metrics.registered_images} / ${metrics.images}`;
    $("point-count").textContent = metrics.exported_points.toLocaleString();
    $("error-count").textContent = `${metrics.after_ba.mean_px.toFixed(3)} px`;
    $("resolution-note").textContent =
      `Errors measured at ${metrics.processing_size.join(" × ")} pixels, before final point filtering.`;
    addResult(
      `${poseLabel}, before BA`,
      metrics.registered_images,
      metrics.points_before_filter,
      metrics.before_ba,
      { mean_track_length: metrics.mean_track_length },
    );
    if (metrics.bundle_adjustment.enabled)
      addResult(
        `${poseLabel} + global BA`,
        metrics.registered_images,
        metrics.points_before_filter,
        metrics.after_ba,
        metrics,
      );
    const ba = metrics.bundle_adjustment;
    const baStatus = ba.enabled
      ? `Bundle adjustment ${ba.converged ? "converged" : "stopped"} after ${ba.evaluations} evaluations.`
      : "Bundle adjustment was disabled.";
    $("ba-note").textContent =
      `${baStatus} ${metrics.exported_points.toLocaleString()} points passed the final reprojection filter. Mean feature track length ${metrics.mean_track_length.toFixed(2)} views. ${(metrics.pairwise_consistency.fraction * 100).toFixed(1)}% of all verified image-pair matches agree with the recovered poses at a 2 px threshold.`;
    setupViewer(scene);
    // Additional dataset and pose-method ablations are exported separately.
    const ablationResponse = await fetch("assets/ablations.json");
    if (ablationResponse.ok) {
      for (const run of await ablationResponse.json())
        addResult(run.name, run.cameras, run.points, run.stats, run);
    }
  } catch (error) {
    status.hidden = false;
    status.textContent = error.message;
  }
}

load();
