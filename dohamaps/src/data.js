
import * as tf from "@tensorflow/tfjs-node";
import * as osPath from "path";
import * as assert from "assert";
import * as glob from "glob";

import * as io from "./io";
import * as image from "./image";

export class Dataset
{
    constructor(histLen, predLen, mode, dim)
    {
        this.clipLen = histLen + predLen;
        this.histLen = histLen;
        this.predLen = predLen;
        this.mode = mode;
        assert.deepEqual(dim.length, 2);
        this.height = dim[0];
        this.width = dim[1];
        this.dataset = null;
    }
    globFiles(path, pattern)
    {
        this.dataset = tf.data.array(glob.sync(osPath.join(path, pattern)))
    }
    batch(batchSize)
    {
        this.dataset = this.dataset.batch(batchSize, false);
    }
    async loadTensors()
    {
        async function pathToTensor(path)
        {
            try { return image.npyToTensor(await io.loadFile(path)); }
            catch (error) { console.log(error); }
        }
        try { this.dataset = await this.dataset.mapAsync(pathToTensor); }
        catch (error) { console.log(error) }
    }
    prefetch()
    {
        this.dataset = this.dataset.prefetch(1);
    }
    tfDataset()
    {
        return this.dataset;
    }
};
