
import * as tf from "@tensorflow/tfjs-node";

import * as layers from "./layers";
import * as losses from "./losses";
import * as metrics from "./metrics";
import * as util from "./util";

function scaleModel(layers, name)
{
    return tf.sequential({ layers: layers, name: name });
}

class Discriminator extends tf.LayersModel
{
    constructor(args)
    {
        const height = args.dimensions[0];
        const width = args.dimensions[1];
        const channels = args.channels;
        const histLen = args.histLen;
        const predLen = args.predLen;
        const numScales = args.numScales;
        const clipLen = histLen + predLen;
        const depth = clipLen * channels;

        const inputs =
        [
            tf.input({ shape: [ util.scale(height, 0, numScales),
                       util.scale(width, 0, numScales), depth ] }),
            tf.input({ shape: [ util.scale(height, 1, numScales),
                       util.scale(width, 1, numScales), depth ] }),
            tf.input({ shape: [ util.scale(height, 2, numScales),
                       util.scale(width, 2, numScales), depth ] }),
            tf.input({ shape: [ util.scale(height, 3, numScales),
                       util.scale(width, 3, numScales), depth ] }),
        ];

        const scale0 =
        [
            layers.input({ shape: inputs[0].shape }),
            layers.conv2d({ filters: 64, kernelSize: 3, padding: "valid" }),
            layers.maxPooling2d({  }),
            layers.flatten({  }),
            layers.dense({ units: 512 }),
            layers.dense({ units: 256 }),
            layers.dense({ units: 1, activation: "sigmoid" }),
        ];

        const scale1 =
        [
            layers.input({ shape: inputs[1].shape }),
            layers.conv2d({ filters: 64, kernelSize: 3, padding: "valid" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "valid" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "valid" }),
            layers.maxPooling2d({  }),
            layers.flatten({  }),
            layers.dense({ units: 1024 }),
            layers.dense({ units: 512 }),
            layers.dense({ units: 1, activation: "sigmoid" }),
        ];

        const scale2 =
        [
            layers.input({ shape: inputs[2].shape }),
            layers.conv2d({ filters: 128, kernelSize: 5, padding: "valid" }),
            layers.conv2d({ filters: 256, kernelSize: 5, padding: "valid" }),
            layers.conv2d({ filters: 256, kernelSize: 5, padding: "valid" }),
            layers.maxPooling2d({  }),
            layers.flatten({  }),
            layers.dense({ units: 1024 }),
            layers.dense({ units: 512 }),
            layers.dense({ units: 1, activation: "sigmoid" }),
        ];

        const scale3 =
        [
            layers.input({ shape: inputs[3].shape }),
            layers.conv2d({ filters: 128, kernelSize: 7, padding: "valid" }),
            layers.conv2d({ filters: 256, kernelSize: 7, padding: "valid" }),
            layers.conv2d({ filters: 512, kernelSize: 5, padding: "valid" }),
            layers.conv2d({ filters: 128, kernelSize: 5, padding: "valid" }),
            layers.maxPooling2d({  }),
            layers.flatten({  }),
            layers.dense({ units: 1024 }),
            layers.dense({ units: 512 }),
            layers.dense({ units: 1, activation: "sigmoid" }),
        ];

        const blocks =
        [
            scaleModel(scale0, "0"),
            scaleModel(scale1, "1"),
            scaleModel(scale2, "2"),
            scaleModel(scale3, "3"),
        ];

        const outputs = [  ];

        for (let i = 0; i < inputs.length; ++i)
            outputs.push(blocks[i].apply(inputs[i]));

        const modelConfig =
        {
            inputs: inputs,
            outputs: outputs,
            name: "Discriminator"
        };

        super(modelConfig);

        this.blocks = blocks;
        this.numScales = numScales;
    }
    call(inputs, kwargs)
    {
        this.invokeCallHook(inputs, kwargs);
        let blocks = this.blocks;
        let numScales = this.numScales;
        function tidy()
        {
            const preds = [  ];
            for (let i = 0; i < numScales; ++i)
                preds.push(tf.clipByValue(blocks[i].apply(inputs[i]), 0.01, 0.99));
            return preds;
        }
        return tf.tidy(tidy);
    }
    static get className()
    {
        return "Discriminator";
    }
    dispose()
    {
        super.dispose();
    }
};

class Generator extends tf.LayersModel
{
    constructor(args, discriminator)
    {
        const height = args.dimensions[0];
        const width = args.dimensions[1];
        const channels = args.channels;
        const histLen = args.histLen;
        const predLen = args.predLen;
        const numScales = args.numScales;
        const histDepth = histLen * channels;
        const predDepth = predLen * channels;

        const concat = layers.concat(-1);
        const upSample = layers.upSampling2d({  });

        const inputs =
        [
            tf.input({ shape: [ util.scale(height, 0, numScales),
                       util.scale(width, 0, numScales), histDepth ] }),
            tf.input({ shape: [ util.scale(height, 1, numScales),
                       util.scale(width, 1, numScales), histDepth ] }),
            tf.input({ shape: [ util.scale(height, 2, numScales),
                       util.scale(width, 2, numScales), histDepth ] }),
            tf.input({ shape: [ util.scale(height, 3, numScales),
                       util.scale(width, 3, numScales), histDepth ] }),
        ];

        const layerInputs =
        [
            tf.input({ shape: [ util.scale(height, 0, numScales),
                       util.scale(width, 0, numScales), histDepth ] }),
            tf.input({ shape: [ util.scale(height, 1, numScales),
                       util.scale(width, 1, numScales), histDepth + predDepth ] }),
            tf.input({ shape: [ util.scale(height, 2, numScales),
                       util.scale(width, 2, numScales), histDepth + predDepth ] }),
            tf.input({ shape: [ util.scale(height, 3, numScales),
                       util.scale(width, 3, numScales), histDepth + predDepth ] }),
        ];

        const scale0 =
        [
            layers.input({ shape: layerInputs[0].shape }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: channels * predLen, kernelSize: 3,
                            padding: "same", activation: "tanh" }),
        ];

        const scale1 =
        [
            layers.input({ shape: layerInputs[1].shape }),
            layers.conv2d({ filters: 128, kernelSize: 5, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: channels * predLen, kernelSize: 5,
                            padding: "same", activation: "tanh" }),
        ];

        const scale2 =
        [
            layers.input({ shape: layerInputs[2].shape }),
            layers.conv2d({ filters: 128, kernelSize: 5, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 512, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: channels * predLen, kernelSize: 5,
                            padding: "same", activation: "tanh" }),
        ];

        const scale3 =
        [
            layers.input({ shape: layerInputs[3].shape }),
            layers.conv2d({ filters: 128, kernelSize: 7, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 512, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: channels * predLen, kernelSize: 7,
                            padding: "same", activation: "tanh" }),
        ];

        const blocks =
        [
            scaleModel(scale0, "0"),
            scaleModel(scale1, "1"),
            scaleModel(scale2, "2"),
            scaleModel(scale3, "3"),
        ];

        const preds = [  ];

        for (let i = 0; i < inputs.length; ++i)
        {
            if (i > 0)
            {
                const upSampled = upSample.apply(preds[i - 1]);
                preds.push(blocks[i].apply(concat.apply([ inputs[i], upSampled ])));
            }
            else
                preds.push(blocks[i].apply(inputs[i]));
        }

        const discInput = [  ]
        for (let i = 0; i < inputs.length; ++i)
            discInput.push(concat.apply([ inputs[i], preds[i] ]));
        const labels = discriminator.apply(discInput);

        const outputs = preds;

        const modelConfig =
        {
            inputs: inputs,
            outputs: outputs,
            name: "Generator"
        };
        super(modelConfig);

        this.blocks = blocks;
        this.numScales = numScales;
        this.discriminator = discriminator;
        this.upSample = upSample;
        this.concat = concat;
        this.labels = labels;
    }
    call(inputs, kwargs)
    {
        this.invokeCallHook(inputs, kwargs);
        let blocks = this.blocks;
        let numScales = this.numScales;
        let discriminator = this.discriminator;
        let upSample = this.upSample;
        let concat = this.concat;
        discriminator.trainable = false;
        function tidy()
        {
            const preds = [  ];
            var scalePred = null;
            for (let i = 0; i < numScales; ++i)
            {
                if (i > 0)
                {
                    scalePred = upSample.apply(scalePred);
                    scalePred = concat.apply([ inputs[i], scalePred ]);
                }
                else
                    scalePred = inputs[i];
                scalePred = blocks[i].apply(scalePred);
                preds.push(scalePred);
            }
            const discInput = [  ];
            for (let i = 0; i < numScales; ++i)
                discInput.push(concat.apply([ inputs[i], preds[i] ]));
            const labels = discriminator.predictOnBatch(discInput);
            return [ preds, labels ];
        }
        const [ preds, labels ] = tf.tidy(tidy);
        this.labels = labels;
        return preds;
    }
    static get className()
    {
        return "Generator";
    }
    dispose()
    {
        super.dispose();
    }
};

export function discriminator(args)
{
    return new Discriminator(args);
}
export function generator(args, discriminator)
{
    return new Generator(args, discriminator);
}

class Combined
{
    constructor(args)
    {
        this.histLen = args.histLen;
        this.predLen = args.predLen;
        this.clipLen = args.histLen + args.predLen;
        this.numScales = args.numScales;
        this.height = args.dimensions[0];
        this.width = args.dimensions[1];
        this.channels = args.channels;

        this.discLearnRate = args.discLearnRate;
        this.genLearnRate = args.genLearnRate;

        this.isTraining = false;

        console.log("  ℹ️   " + tf.memory().numTensors + " tensors");
        console.log("  ℹ️   initializing discriminator...");
        this.discriminator = discriminator(args);
        console.log("  ℹ️   discriminator initialized");
        console.log("  ℹ️   initializing generator...");
        this.generator = generator(args, this.discriminator);
        console.log("  ℹ️   generator initialized");
        console.log("  ℹ️   " + tf.memory().numTensors + " tensors");
    }
    compile()
    {
        console.log("  ℹ️   compiling model...");
        const discConfig =
        {
            optimizer: tf.train.adam(this.discLearnRate),
            loss: losses.adversarial
        };
        this.discriminator.compile(discConfig);
        const genConfig =
        {
            optimizer: tf.train.adam(this.genLearnRate),
            loss: losses.combined
        };
        this.generator.compile(genConfig);
        this.metrics =
        [
            metrics.psnr,
            metrics.sharpdiff,
        ];
        console.log("  ℹ️   model compiled");
    }
    recompile(inputs)
    {
        this.generator.apply(inputs);
        let labels = tf.mean(tf.stack(this.generator.labels), 0);
        const genConfig =
        {
            optimizer: tf.train.adam(this.genLearnRate),
            loss: function (yTrue, yPred)
                  {
                      return losses.combined(yTrue, yPred, labels);
                  }
        };
        this.generator.compile(genConfig);
    }
    async trainStep(data)
    {
        let generator = this.generator;
        let numScales = this.numScales;
        function tidy()
        {
            console.log("  ℹ️   beginning training step...");
            const histScales = data[0];
            const gtScales = data[1];

            const fgScales = generator.predictOnBatch(histScales);

            const batchSize = fgScales[0].shape[0];

            const discInput = [  ];
            for (let i = 0; i < numScales; ++i)
            {
                var fakeSequence = tf.concat([ histScales[i], fgScales[i] ], -1);
                var realSequence = tf.concat([ histScales[i], gtScales[i] ], -1);
                discInput.push(tf.concat([ fakeSequence, realSequence ], 0));
            }

            const labelShape = [ batchSize, 1 ];
            var discLabel = tf.concat([ tf.zeros(labelShape), tf.ones(labelShape) ], 0);
            const noise = tf.mul(tf.scalar(0.05), tf.randomUniform(discLabel.shape));
            discLabel = tf.add(discLabel, noise);

            const discLabels = [  ];
            for (let i = 0; i < numScales; ++i)
                discLabels.push(discLabel);
            return [ discInput, discLabels, histScales, gtScales ];
        }

        const [ discInput, discLabels, histScales, gtScales ] = tf.tidy(tidy);

        /*** discriminator ***/

        console.log("  ℹ️   training discriminator...");
        const discLoss = await this.discriminator.trainOnBatch(discInput, discLabels);

        /*** generator ***/

        console.log("  ℹ️   training generator...");
        this.recompile(histScales);
        const genLoss = await this.generator.trainOnBatch(histScales, gtScales);

        /*** output ***/

        const out =
        [
            { name: "Discriminator loss", value: discLoss },
            { name: "Generator loss", value: genLoss },
        ];

        return out;
    }
    async fit(dataset, epochs)
    {
        if (this.isTraining)
            throw new Error("model is already training");
        this.isTraining = true;
        for (let epoch = 0; epoch < epochs; ++epoch)
        {

        }
    }
    dispose()
    {
        console.log("  ℹ️   " + tf.memory().numTensors + " tensors");
        this.discriminator.dispose();
        this.generator.dispose();
        console.log("  ℹ️   " + tf.memory().numTensors + " tensors");
    }
};

export function combined(args)
{
    return new Combined(args);
}
