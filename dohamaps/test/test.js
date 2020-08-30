
import * as tf from "@tensorflow/tfjs-node";
import * as dohamaps from "../";

import * as osPath from "path";

class Tester
{
    constructor(debug)
    {
        if (debug)
            tf.enableDebugMode();
    }
    async processClips()
    {
        try
        {
            const loadPath = osPath.join(__dirname, "../../data");
            const savePath = osPath.join(__dirname, "../../clips");
            dohamaps.processClips(10, 4, loadPath, savePath, [ 64, 64 ]);
        }
        catch (error) { console.error(error); }
    }
    async dataset()
    {
        try
        {
            const config =
            {
                histLen: 2,
                predLen: 2,
                numScales: 4,
                dimensions: [ 64, 64 ],
                channels: 3,
            }
            const dataset = dohamaps.data.dataset(config);
            const clipsPath = osPath.join(__dirname, "../../clips");
            await dataset.globFiles(clipsPath, "*.tns");
            await dataset.loadTensors();
            dataset.batch(2);
            dataset.splitTensorsTrain();
            dataset.scale();
            const array = await dataset.backend.toArray();
            for (let each of array)
                console.log(each);
        }
        catch (error) { console.error(error); }
    }
    async combined()
    {
        try
        {
            const config =
            {
                histLen: 2,
                predLen: 2,
                numScales: 4,
                dimensions: [ 64, 64 ],
                channels: 3,
                discLearnRate: 0.1,
                genLearnRate: 0.1,
            }
            const dataset = dohamaps.data.dataset(config);
            const clipsPath = osPath.join(__dirname, "../../clips");
            await dataset.globFiles(clipsPath, "*.tns");
            await dataset.loadTensors();
            dataset.batch(2);
            dataset.splitTensorsTrain();
            dataset.scale();
            const gan = dohamaps.models.combined(config);
            gan.compile();
            await gan.fit(dataset, 5);
            gan.dispose();
        }
        catch (error) { console.error(error); }
    }
    static profile(info)
    {
        if (info.newTensors > 0)
            console.log("  ℹ️   " + info.newTensors + " undisposed tensors");
    }
}

async function test()
{
    try
    {
        const tester = new Tester();
        // await tester.processClips();
        // await tester.dataset();
        await tester.combined();
    }
    catch (error) { console.error(error); }
}

tf.profile(test).then(Tester.profile);
