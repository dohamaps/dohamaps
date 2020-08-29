
import * as tf from "@tensorflow/tfjs-node";
import * as assert from "assert";

import * as io from "./io";
import * as image from "./image";
import * as util from "./util";

class Dataset
{
    constructor(args)
    {
        this.histLen = args.histLen;
        this.predLen = args.predLen;
        this.clipLen = this.histLen + this.predLen;
        this.numScales = args.numScales;
        this.dimensions = args.dimensions;
        assert.deepEqual(this.dimensions.length, 2);
        this.height = this.dimensions[0];
        this.width = this.dimensions[1];
        this.channels = args.channels;

        this.backend = null;
    }
    async globFiles(path, pattern)
    {
        this.backend = tf.data.array(await util.glob(path, pattern));
    }
    batch(batchSize)
    {
        this.backend = this.backend.batch(batchSize, false);
    }
    async loadTensors()
    {
        async function map(path)
        {
            try { return util.tnsToTensor(await io.loadFile(path)); }
            catch (error) { console.log(error); }
        }
        this.backend = this.backend.mapAsync(map);
    }
    splitTensorsTrain()
    {
        let histLen = this.histLen;
        let predLen = this.predLen;
        let channels = this.channels;
        function map(tensor)
        {
            const shape = tensor.shape;
            const histStart = [ 0, 0, 0, 0 ];
            const histSize = [ -1, -1, -1, histLen * channels ];
            const predStart = [ 0, 0, 0, histLen * channels ];
            const predSize = [ -1, -1, -1, predLen * channels ];

            const history = tensor.slice(histStart, histSize);
            const groundTruth = tensor.slice(predStart, predSize);
            return [ history, groundTruth ];
        }
        this.backend = this.backend.map(map);
    }
    scale()
    {
        let numScales = this.numScales;
        function map(pair)
        {
            const history = pair[0];
            const groundTruth = pair[1];

            const histScales = [  ];
            const gtScales = [  ];

            for (let i = 0; i < numScales; ++i)
            {
                histScales.push(image.resize(history, i, numScales));
                gtScales.push(image.resize(groundTruth, i, numScales));
            }
            return [ histScales, gtScales ];
        }
        this.backend = this.backend.map(map);
    }
    prefetch()
    {
        this.backend = this.backend.prefetch(1);
    }
};

export function dataset(args)
{
    return new Dataset(args);
}
