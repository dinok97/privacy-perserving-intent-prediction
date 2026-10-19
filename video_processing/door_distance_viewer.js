/* A single assumption-based scale is applied to the entire reconstructed scene. */
(function (root) {
  'use strict';
  function median(values) {
    const a = [...values].sort((x, y) => x - y);
    if (!a.length) throw new Error('No reference door widths available.');
    const middle = Math.floor(a.length / 2);
    return a.length % 2 ? a[middle] : (a[middle - 1] + a[middle]) / 2;
  }
  function estimateScale(calibration, assumedWidth, reference) {
    if (!Number.isFinite(assumedWidth) || assumedWidth <= 0) {
      throw new Error('Enter a positive door-leaf width in metres.');
    }
    const candidates = calibration.candidates.filter(c => reference === 'all_median' ||
      (reference === 'uncropped_median' ? c.included_by_default : c.door_id === reference));
    const factors = candidates.map(c => {
      if (!Number.isFinite(c.width_model_units) || c.width_model_units <= 0) {
        throw new Error('A reconstructed door width is invalid.');
      }
      return assumedWidth / c.width_model_units;
    });
    return { factor: median(factors), candidates, factors, assumedWidth, reference };
  }
  const scaleValue = (value, factor) => value === null ? null : value * factor;
  const scaleCoordinates = (values, factor) => values.map(v => scaleValue(v, factor));
  const api = { median, estimateScale, scaleValue, scaleCoordinates };
  root.DoorMeasurements = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;

  function start() {
    const data = JSON.parse(document.getElementById('measurement-data').textContent);
    const slider = document.getElementById('frame');
    const units = document.getElementById('measurement-units');
    const width = document.getElementById('door-width');
    const reference = document.getElementById('scale-reference');
    const play = document.getElementById('play');
    let timer = null;
    let displayFactor = 1;
    let lastUnit = 'model';
    let currentScale = estimateScale(data.calibration, Number(width.value), reference.value);

    function table(row) {
      document.getElementById('values').replaceChildren();
      data.doors.forEach(door => {
        const tr = document.createElement('tr');
        const name = document.createElement('td');
        name.textContent = door.name;
        name.style.color = door.colour;
        tr.append(name);
        [row.distances[door.id], row.closure_rates[door.id]].forEach(value => {
          const td = document.createElement('td');
          td.textContent = value === null ? '—' : (value * displayFactor).toFixed(4);
          tr.append(td);
        });
        document.getElementById('values').append(tr);
      });
    }
    function show(index) {
      const row = data.rows[index];
      slider.value = index;
      document.getElementById('image').src = 'frames/frame_' + String(row.frame_index).padStart(6, '0') + '.jpg';
      document.getElementById('time').textContent = row.frame_index + ' / ' + (data.rows.length - 1) +
        ' — ' + row.t.toFixed(2) + ' s';
      document.getElementById('status').textContent = row.valid_3d ?
        'Track ' + row.track_id + ': ' + row.contact_source.replaceAll('_', ' ') +
          (row.contact_on_floor_label ? '' : ' (outside segmented floor; plane extrapolation)') :
        (row.detected ? 'Person detected; feet are cropped or unobserved. 3D distance unavailable.' :
          'No person detected. 3D distance unavailable.');
      table(row);
      const p = row.point_ground;
      Plotly.restyle('scene', {
        x: [p ? [p[0] * displayFactor] : []],
        y: [p ? [p[1] * displayFactor] : []],
        z: [p ? [p[2] * displayFactor] : []]
      }, [data.markerIndex]);
      Plotly.relayout('curves', { 'shapes[0].x0': row.t, 'shapes[0].x1': row.t });
    }
    function scaleEvidence() {
      const scale = currentScale;
      const selected = scale.candidates.map(c => c.door_id);
      const names = scale.candidates.map(c => c.name).join(', ');
      document.getElementById('scale-note').textContent =
        'Assumed leaf width: ' + (scale.assumedWidth * 100).toFixed(1) + ' cm. ' +
        '1 model unit ≈ ' + scale.factor.toFixed(4) + ' m. Reference: ' + names +
        (scale.candidates.some(c => c.boundary_cropped) ? ' (includes a cropped panel).' : '.');
      document.getElementById('scale-evidence').replaceChildren();
      data.calibration.candidates.forEach(c => {
        const tr = document.createElement('tr');
        [c.name, c.width_model_units.toFixed(4), (scale.assumedWidth / c.width_model_units).toFixed(4),
          (c.width_model_units * scale.factor).toFixed(3), selected.includes(c.door_id) ? 'Used' : 'Excluded']
          .forEach(value => {
            const td = document.createElement('td');
            td.textContent = value;
            tr.append(td);
          });
        document.getElementById('scale-evidence').append(tr);
      });
      const range = [Math.min(...scale.factors), Math.max(...scale.factors)];
      document.getElementById('scale-disagreement').textContent = scale.factors.length > 1 ?
        'The selected doors imply ' + range[0].toFixed(3) + '–' + range[1].toFixed(3) +
        ' metres per model unit. This spread is disagreement between estimates, not a confidence interval.' :
        'A single-door reference provides no comparison between door-based scales.';
    }
    function applyUnits() {
      try {
        currentScale = estimateScale(data.calibration, Number(width.value), reference.value);
      } catch (error) {
        document.getElementById('scale-error').textContent = error.message;
        width.setAttribute('aria-invalid', 'true');
        units.value = lastUnit;
        return;
      }
      width.removeAttribute('aria-invalid');
      document.getElementById('scale-error').textContent = '';
      lastUnit = units.value;
      const metric = units.value === 'metres';
      displayFactor = metric ? currentScale.factor : 1;
      const unit = metric ? 'estimated m' : 'model units';
      document.getElementById('distance-heading').textContent = '3D distance (' + unit + ')';
      document.getElementById('rate-heading').textContent = 'Change (' + (metric ? 'estimated m/s' : 'model units/s') + ')';
      document.getElementById('unit-description').textContent = metric ?
        'Estimated metres from the assumed door-leaf width. This is not a measured calibration.' :
        'Original LingBot model units. These 3D units are not image pixels.';
      scaleEvidence();
      // Use immutable numeric coordinates embedded at export time, never scale already-scaled data.
      const coordinates = data.sceneCoordinates;
      Plotly.restyle('scene', {
        x: coordinates.map(c => scaleCoordinates(c.x, displayFactor)),
        y: coordinates.map(c => scaleCoordinates(c.y, displayFactor)),
        z: coordinates.map(c => scaleCoordinates(c.z, displayFactor))
      }, coordinates.map((_, index) => index));
      Plotly.relayout('scene', {
        'scene.xaxis.title.text': 'Right (' + unit + ')',
        'scene.yaxis.title.text': 'Forward (' + unit + ')',
        'scene.zaxis.title.text': 'Height above fitted floor (' + unit + ')',
        'scene.xaxis.autorange': true, 'scene.yaxis.autorange': true, 'scene.zaxis.autorange': true
      });
      Plotly.restyle('curves', {
        y: data.doors.map(door => data.rows.map(row => scaleValue(row.distances[door.id], displayFactor))),
        hovertemplate: data.doors.map(() => '%{x:.2f}s: %{y:.4f} ' + unit + '<extra>%{fullData.name}</extra>')
      }, data.doors.map((_, index) => index));
      Plotly.relayout('curves', { 'yaxis.title.text': 'Distance (' + unit + ')', 'yaxis.autorange': true });
      show(Number(slider.value));
    }
    slider.addEventListener('input', () => show(Number(slider.value)));
    units.addEventListener('change', applyUnits);
    width.addEventListener('change', applyUnits);
    reference.addEventListener('change', applyUnits);
    play.addEventListener('click', () => {
      if (timer) {
        clearInterval(timer); timer = null; play.textContent = 'Play'; return;
      }
      if (Number(slider.value) === data.rows.length - 1) show(0);
      play.textContent = 'Pause';
      timer = setInterval(() => {
        const next = Number(slider.value) + 1;
        if (next >= data.rows.length) {
          clearInterval(timer); timer = null; play.textContent = 'Play'; return;
        }
        show(next);
      }, 1000 / data.fps);
    });
    document.getElementById('download-current').addEventListener('click', () => {
      let calibration;
      try { calibration = estimateScale(data.calibration, Number(width.value), reference.value); }
      catch (_) { applyUnits(); return; }
      const fields = ['frame_index', 't', 'track_id', 'valid_3d', 'assumed_door_leaf_width_m',
        'metres_per_model_unit', 'scale_reference'];
      data.doors.forEach(d => fields.push('distance_3d_' + d.id + '_model_units',
        'distance_3d_' + d.id + '_estimated_m', 'closure_rate_3d_' + d.id + '_model_units_per_s',
        'closure_rate_3d_' + d.id + '_estimated_m_per_s'));
      const csv = [fields.join(',')];
      data.rows.forEach(row => {
        const values = [row.frame_index, row.t, row.track_id, row.valid_3d,
          calibration.assumedWidth, calibration.factor, calibration.reference];
        data.doors.forEach(d => values.push(row.distances[d.id], scaleValue(row.distances[d.id], calibration.factor),
          row.closure_rates[d.id], scaleValue(row.closure_rates[d.id], calibration.factor)));
        csv.push(values.map(v => v === null ? '' : String(v)).join(','));
      });
      const url = URL.createObjectURL(new Blob([csv.join('\n') + '\n'], { type: 'text/csv;charset=utf-8' }));
      const link = document.createElement('a');
      link.href = url; link.download = data.videoStem + '_distances_current_calibration.csv'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    applyUnits();
  }
  function ready() {
    Promise.all([root.sceneReady, root.curvesReady]).then(start).catch(error => {
      document.getElementById('scale-error').textContent = 'Could not initialize plots: ' + error.message;
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready);
  else ready();
})(globalThis);
