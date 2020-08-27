
import * as tf from "@tensorflow/tfjs-node";

export class Discriminator extends tf.LayersModel
{
    constructor()
    {
        super.constructor();

        disc0Layers =
        [
            layers.conv2d({ filters: 64, kernelSize: 3, padding: "valid" }),
            layers.maxPooling2d({  }),
            layers.flatten({  }),
            layers.dense({ units: 512 }),
            layers.dense({ units: 256 }),
            layers.dense({ units: 1, activation: "sigmoid" }),
        ];

        disc1Layers =
        [
            layers.conv2d({ filters: 64, kernelSize: 3, padding: "valid" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "valid" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "valid" }),
            layers.maxPooling2d({  }),
            layers.flatten({  }),
            layers.dense({ units: 1024 }),
            layers.dense({ units: 512 }),
            layers.dense({ units: 1, activation: "sigmoid" }),
        ];

        disc2Layers =
        [
            layers.conv2d({ filters: 128, kernelSize: 5, padding: "valid" }),
            layers.conv2d({ filters: 256, kernelSize: 5, padding: "valid" }),
            layers.conv2d({ filters: 256, kernelSize: 5, padding: "valid" }),
            layers.maxPooling2d({  }),
            layers.flatten({  }),
            layers.dense({ units: 1024 }),
            layers.dense({ units: 512 }),
            layers.dense({ units: 1, activation: "sigmoid" }),
        ];

        disc3Layers =
        [
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

        this.blocks =
        [
            layers.block(disc0Layers),
            layers.block(disc1Layers),
            layers.block(disc2Layers),
            layers.block(disc3Layers),
        ];

        this.numScales = this.blocks.length;
    }
    apply(inputs)
    {
        var preds = [  ];
        for (let i = 0; i < this.numScales; ++i)
            preds.push(tf.clipByValue(block[i].apply(inputs[i]), 0.01, 0.99));
        return preds;
    }
};

export class Generator extends tf.LayersModel
{
    constructor(channels, predLen)
    {
        super.constructor();

        this.channels = channels;
        this.predLen = predLen;

        gen0Layers =
        [
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: this.channels * this.predLen, kernelSize: 3,
                            padding: "same", activation: "tanh" }),
        ];

        gen1Layers =
        [
            layers.conv2d({ filters: 128, kernelSize: 5, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: this.channels * this.predLen, kernelSize: 5,
                            padding: "same", activation: "tanh" }),
        ];

        gen2Layers =
        [
            layers.conv2d({ filters: 128, kernelSize: 5, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 512, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: this.channels * this.predLen, kernelSize: 5,
                            padding: "same", activation: "tanh" }),
        ];

        gen3Layers =
        [
            layers.conv2d({ filters: 128, kernelSize: 7, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 512, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 256, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: 128, kernelSize: 3, padding: "same" }),
            layers.conv2d({ filters: this.channels * this.predLen, kernelSize: 7,
                            padding: "same", activation: "tanh" }),
        ];

        this.blocks =
        [
            layers.block(gen0Layers),
            layers.block(gen1Layers),
            layers.block(gen2Layers),
            layers.block(gen3Layers),
        ];

        this.numScales = this.blocks.length;
        this.upSample = layers.upSampling2d({  });
    }
    apply(inputs)
    {
        var preds = [  ];
        var scalePred;
        for (let i = 0; i < this.numScales; ++i)
        {
            if (i > 0)
            {
                scalePred = this.upSample.apply(scalePred);
                scalePred = tf.concat([ inputs[i], scalePred ], axis = 3);
            }
            else
                scalePred = inputs[i];
            scalePred = blocks[i].apply(scalePred);
            preds.push(scalePred);
        }
        return preds;
    }
};
