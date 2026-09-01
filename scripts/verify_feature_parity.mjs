import { melSpectrogramFromSamples } from '../mobile/featureExtraction.js';
import fs from 'fs';

const ref = JSON.parse(fs.readFileSync('./ref_data.json', 'utf8'));
const samples = Float32Array.from(ref.samples);
const refMel = ref.mel; // [64][101]

const jsMel = melSpectrogramFromSamples(samples); // [64][nFrames]

console.log('Python ref shape:', refMel.length, 'x', refMel[0].length);
console.log('JS shape:', jsMel.length, 'x', jsMel[0].length);

let maxAbsDiff = 0, sumAbsDiff = 0, count = 0;
for (let m = 0; m < refMel.length; m++) {
  for (let f = 0; f < refMel[0].length; f++) {
    const diff = Math.abs(refMel[m][f] - jsMel[m][f]);
    maxAbsDiff = Math.max(maxAbsDiff, diff);
    sumAbsDiff += diff;
    count++;
  }
}
console.log('Max abs diff (dB):', maxAbsDiff.toFixed(4));
console.log('Mean abs diff (dB):', (sumAbsDiff / count).toFixed(4));
console.log('Sample ref[0][0:5]:', refMel[0].slice(0, 5));
console.log('Sample js[0][0:5]:', Array.from(jsMel[0].slice(0, 5)));
