
import * as tf from "@tensorflow/tfjs-node";

import * as initializers from "./initializers";

export function conv2d(options)
{
    const config =
    {
        filters: options.filters,
        kernelSize: options.kernelSize,
        strides: 1,
        padding: options.padding,
        dataFormat: "channelsLast",
        activation: options.activation ? options.activation : "relu",
        useBias: true,
        kernelInitializer: initializers.truncatedNormal(),
        biasInitializer: initializers.constant(),
    };
    return tf.layers.conv2d(config);
}

export function maxPooling2d(options)
{
    const config =
    {
        poolSize: options.poolSize ? options.poolSize : 2,
        strides: 2,
        padding: "valid",
        dataFormat: "channelsLast",
    };
    return tf.layers.maxPooling2d(config);
}

export function flatten(options)
{
    const config = { dataFormat: "channelsLast" };
    return tf.layers.flatten(config);
}

export function dense(options)
{
    const config =
    {
        units: options.units,
        activation: options.activation ? options.activation : "relu",
        useBias: true,
        kernelInitializer: initializers.truncatedNormal(),
        biasInitializer: initializers.constant(),
    }
    return tf.layers.dense(config);
}

export function upSampling2d(options)
{
    const config =
    {
        size: options.size ? options.size : [ 2, 2 ],
        dataFormat: "channelsLast",
    }
    return tf.layers.upSampling2d(config);
}

class Block extends tf.layers.Layer
{
    constructor(layerList, name)
    {
        super({ name: "Block_" + name });
        this.layerList = layerList;
    }
    call(inputs, kwargs)
    {
        this.invokeCallHook(inputs, kwargs);
        let layerList = this.layerList;
        function tidy()
        {
            var outputs = inputs;
            for (let layer of layerList)
                outputs = layer.apply(outputs);
            return outputs;
        }
        return tf.tidy("layers.Block.apply", tidy);
    }
    computeOutputShape(inputShape)
    {
        let layerList = this.layerList;
        function tidy()
        {
            var outputShape = inputShape;
            for (let layer of layerList)
                outputShape = layer.computeOutputShape(outputShape);
            return outputShape;
        }
        return tf.tidy("layers.Block.computeOutputShape", tidy);
    }
    static get className()
    {
        return "Block";
    }
};

export function block(layerList, name)
{
    return new Block(layerList, name);
}
