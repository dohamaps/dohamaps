
import * as tf from "@tensorflow/tfjs-node";
import * as dohamaps from "../";

class Tester
{
    constructor()
    {
        // tf.enableDebugMode();
    }
    async saveTns()
    {
        try
        {
            var buffer = await dohamaps.io.loadFile("/Users/admin/Desktop/frong.png");
            const tensor = dohamaps.image.imageToTensor(buffer);
            const ops =
            [
                function(tensor) { return dohamaps.image.cropRandom(tensor, [ 64, 64 ]); },
                function(tensor) { return dohamaps.image.normalize(tensor); },
                async function(tensor) { return await dohamaps.util.tensorToTns(tensor); }
            ];
            buffer = await dohamaps.memory.chain(ops, tensor);
            tf.dispose(tensor);
            await dohamaps.io.saveFile("/Users/admin/Desktop/", "fring.tns", buffer);
        }
        catch (error) { console.log(error); }
    }
    async saveImage()
    {
        try
        {
            var buffer = await dohamaps.io.loadFile("/Users/admin/Desktop/fring.tns");
            const tensor = dohamaps.util.tnsToTensor(buffer);
            const ops =
            [
                function(tensor) { return dohamaps.image.denormalize(tensor); },
                async function(tensor) { return await dohamaps.image.tensorToImage(tensor); }
            ];
            buffer = await dohamaps.memory.chain(ops, tensor);
            tf.dispose(tensor);
            await dohamaps.io.saveFile("/Users/admin/Desktop/", "fring.png", buffer);

            tf.dispose(tensor);
        }
        catch (error) { console.log(error); }
    }
    async processClips()
    {
        try
        {
            dohamaps.processClips(10, 4, "/Users/admin/github/dohamaps/data", "/Users/admin/Desktop/clips", [ 64, 64 ]);
        }
        catch (error) { console.log(error); }
    }
    async losses()
    {
        try
        {
            const bufferFring = await dohamaps.io.loadFile("/Users/admin/Desktop/fring.tns");
            const bufferFrang = await dohamaps.io.loadFile("/Users/admin/Desktop/frang.tns");

            const fring = tf.tidy(function() { return tf.expandDims(dohamaps.util.tnsToTensor(bufferFring)) });
            const frang = tf.tidy(function() { return tf.expandDims(dohamaps.util.tnsToTensor(bufferFrang)) });
            const gdl = dohamaps.losses.gdl([ fring ], [ frang ]);
            console.log("  ℹ️   gdl: " + gdl.toString());
            tf.dispose(fring);
            tf.dispose(frang);
            tf.dispose(gdl);
        }
        catch (error) { console.log(error); }
    }
    async metrics()
    {
        try
        {
            const bufferFring = await dohamaps.io.loadFile("/Users/admin/Desktop/fring.tns");
            const bufferFrang = await dohamaps.io.loadFile("/Users/admin/Desktop/frang.tns");

            const fring = tf.tidy(function() { return tf.expandDims(dohamaps.util.tnsToTensor(bufferFring)) });
            const frang = tf.tidy(function() { return tf.expandDims(dohamaps.util.tnsToTensor(bufferFrang)) });
            const sharpdiff = dohamaps.metrics.sharpdiff(fring, frang);
            console.log("  ℹ️   sharpdiff: " + sharpdiff.toString());
            tf.dispose(fring);
            tf.dispose(frang);
            tf.dispose(sharpdiff);
        }
        catch (error) { console.log(error); }
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
            await dataset.globFiles("/Users/admin/Desktop/clips", "*.tns");
            await dataset.loadTensors();
            dataset.batch(2);
            dataset.splitTensorsTrain();
            dataset.scale();
            const array = await dataset.backend.toArray();
            for (let each of array)
                console.log(each);
        }
        catch (error) { console.log(error); }
    }
    async discriminator()
    {
        try
        {
            const buffer = await dohamaps.io.loadFile("/Users/admin/Desktop/clips/1.tns");
            function tidy()
            {
                const tensor = dohamaps.util.tnsToTensor(buffer);
                const inputs =
                [
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 8, 8 ])),
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 16, 16 ])),
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 32, 32 ])),
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 64, 64 ]))
                ]
                const config =
                {
                    histLen: 2,
                    predLen: 2,
                    numScales: 4,
                    dimensions: [ 64, 64 ],
                    channels: 3,
                }
                const discriminator = dohamaps.models.discriminator(config);
                const outputs = discriminator.apply(inputs);
                for (let output of outputs)
                    console.log(output.toString());
            }
            tf.tidy(tidy);
        }
        catch (error) { console.log(error); }
    }
    async generator()
    {
        try
        {
            const buffer = await dohamaps.io.loadFile("/Users/admin/Desktop/clips/1.tns");
            function tidy()
            {
                const tensor = dohamaps.util.tnsToTensor(buffer);
                const inputs =
                [
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 8, 8 ])),
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 16, 16 ])),
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 32, 32 ])),
                    tf.expandDims(tf.image.resizeBilinear(tensor, [ 64, 64 ]))
                ]
                const config =
                {
                    histLen: 2,
                    predLen: 2,
                    numScales: 4,
                    dimensions: [ 64, 64 ],
                    channels: 3,
                }
                const generator = dohamaps.models.generator(config);
                const outputs = generator.apply(inputs);
                for (let output of outputs)
                    console.log(output.toString());
            }
            tf.tidy(tidy);
        }
        catch (error) { console.log(error); }
    }
    async combined()
    {
        try
        {
            // const buffer = await dohamaps.io.loadFile("/Users/admin/Desktop/clips/1.tns");
            function tidy()
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
                const gan = dohamaps.models.combined(config);
                gan.compile();
            }
            tf.tidy(tidy);
        }
        catch (error) { console.log(error); }
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

        // await tester.saveTns();
        // await tester.saveImage();
        // await tester.losses();
        // await tester.metrics();
        // await tester.processClips();
        await tester.dataset();
        // await tester.discriminator();
        // await tester.generator();
        // await tester.combined();
    }
    catch (error) { console.log(error); }
}

tf.profile(test).then(Tester.profile);
