
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
        inputShape: options.inputShape
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

class Concat extends tf.layers.Layer
{
    constructor(axis, name)
    {
        super({ name: "Concat_" + name });
        this.axis = axis;
    }
    call(inputs, kwargs)
    {
        this.invokeCallHook(inputs, kwargs);
        let axis = this.axis;
        function tidy()
        {
            return tf.concat(inputs, axis);
        }
        return tf.tidy("layers.Concat.apply", tidy);
    }
    computeOutputShape(inputShapes)
    {
        let axis = this.axis;
        function tidy()
        {
            var outputShape = null;
            for (let shape of inputShapes)
            {
                if (!outputShape)
                    outputShape = shape.slice();
                else
                    if (axis < 0)
                        outputShape[shape.length + axis] += shape[shape.length + axis];
                    else
                        outputShape[axis] += shape[axis];
            }
            return outputShape;
        }
        return tf.tidy("layers.Concat.computeOutputShape", tidy);
    }
    static get className()
    {
        return "Concat";
    }
};

class Stack extends tf.layers.Layer
{
    constructor(name)
    {
        super({ name: "Stack_" + name });
        this.axis = axis;
    }
    call(inputs, kwargs)
    {
        this.invokeCallHook(inputs, kwargs);
        function tidy()
        {
            return tf.stack(inputs);
        }
        return tf.tidy("layers.Stack.apply", tidy);
    }
    computeOutputShape(inputShapes)
    {
        function tidy()
        {
            const length = inputShapes.length;
            return [ length ].concat(inputShapes[0].shape);
        }
        return tf.tidy("layers.Stack.computeOutputShape", tidy);
    }
    static get className()
    {
        return "Stack";
    }
};

export function stack(name = "")
{
    return new Stack(name);
}

export function block(layerList, name)
{
    return new Block(layerList, name);
}

export function concat(axis, name = "")
{
    return new Concat(axis, name);
}

export function input(options)
{
    const config =
    {
        batchInputShape: options.shape,
        dtype: "float32",
        sparse: false,
    };
    return tf.layers.inputLayer(config);
}
