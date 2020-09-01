
import * as tf from "@tensorflow/tfjs-node";
import * as osPath from "path";

import * as util from "./util";
import * as image from "./image";
import * as io from "./io";
import * as data from "./data";
import * as layers from "./layers";
import * as models from "./models";
import * as metrics from "./metrics";
import * as losses from "./losses";
import * as memory from "./memory";
import * as initializers from "./initializers";

export async function processClips(numClips, clipLen, loadPath, savePath, dimensions)
{
    async function getClip(path, startIndex = null)
    {
        try
        {
            var imgPaths = (await util.glob(path, "*")).sort();
            var clip = null;
            if (startIndex)
                imgPaths = imgPaths.slice(startIndex, startIndex + clipLen);
            else
            {
                startIndex = util.random(0, imgPaths.length - clipLen);
                imgPaths = imgPaths.slice(startIndex, startIndex + clipLen);
            }
            for (let imgPath of imgPaths)
            {
                const buffer = await io.loadFile(imgPath);
                if (clip)
                    clip = tf.tidy(function() { return clip.concat(image.imageToTensor(buffer), -1); });
                else
                    clip = image.imageToTensor(buffer);
            }
            return clip;
        }
        catch (error) { console.error(error); }
    }
    try
    {
        const savePredPath = osPath.join(savePath, "pred");
        const startIndex = (await util.glob(loadPath, "*")).length - clipLen;
        var clipTensor = await getClip(loadPath, startIndex);
        await io.saveFile(savePredPath, "pred.tns", await util.tensorToTns(clipTensor));

        tf.dispose(clipTensor);

        const numExisting = (await util.glob(savePath, "*.tns")).length;
        for (let i = numExisting; i < numExisting + numClips; ++i)
        {
            clipTensor = image.cropRandom(await getClip(loadPath), dimensions);
            await io.saveFile(savePath, i + ".tns", await util.tensorToTns(clipTensor));

            tf.dispose(clipTensor);
        }
    }
    catch (error) { console.error(error); }
}

export async function train(args)
{
    const config =
    {
        histLen: args.histLen,
        predLen: args.predLen,
        numScales: args.numScales ? args.numScales : 4,
        dimensions: args.trainDimensions,
        channels: args.channels ? args.channels : 3,
        discLearnRate: args.discLearnRate,
        genLearnRate: args.genLearnRate,
    }
    const dataset = data.dataset(config);
    const clipsPath = osPath.join(__dirname, "../../clips");
    await dataset.globFiles(clipsPath, "*.tns");
    await dataset.loadTensors();
    dataset.batch(args.batchSize);
    dataset.splitTensorsTrain();
    dataset.scale();
    const gan = models.combined(config);
    gan.compile();
    await gan.fit(dataset, args.epochs);
    gan.dispose();
}

export
{
    util,
    image,
    io,
    data,
    layers,
    models,
    metrics,
    losses,
    memory,
    initializers
}
