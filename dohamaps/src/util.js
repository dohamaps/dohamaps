
import * as tf from "@tensorflow/tfjs-node";
import * as assert from "assert";
import * as nodeUtil from "util";
import * as globCallback from "glob";
import * as osPath from "path";

export const UNKNOWN_ENDIAN = 0x0;
export const LITTLE_ENDIAN = 0x1;
export const BIG_ENDIAN = 0x2;

export function scale(value, scaleIndex, numScales)
{
    return value / (2.0 ** (numScales - 1)) * (2.0 ** scaleIndex);
}

export async function glob(path, pattern)
{
    const globPromise = nodeUtil.promisify(globCallback.glob);
    return await globPromise(osPath.join(path, pattern));
}

export function toArrayBuffer(view)
{
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

export function toBuffer(arrayBuffer)
{
    return Buffer.from(new Uint8Array(arrayBuffer));
}

export function random(min, max)
{
    return Math.floor(Math.random() * (max - min)) + min;
}

export function endian()
{
    const arrayBuffer = new ArrayBuffer(2);
    const uint8Array = new Uint8Array(arrayBuffer);
    const uint16array = new Uint16Array(arrayBuffer);
    uint8Array[0] = 0xAA;
    uint8Array[1] = 0xBB;
    if (uint16array[0] === 0xBBAA) return LITTLE_ENDIAN;
    if (uint16array[0] === 0xAABB) return BIG_ENDIAN;
    else return UNKNOWN_ENDIAN;
}

export async function tensorToTns(tensor)
{
    assert.deepEqual(tensor.dtype, "float32");
    const rank = Int32Array.from([ tensor.rank ]);
    const shape = Int32Array.from(tensor.shape);
    const data = Float32Array.from(await tensor.data());
    const rankBuffer = Buffer.from(rank.buffer);
    const shapeBuffer = Buffer.from(shape.buffer);
    const dataBuffer = Buffer.from(data.buffer);
    return Buffer.concat([ rankBuffer, shapeBuffer, dataBuffer ]);
}

export function tnsToTensor(buffer)
{
    function tidy()
    {
        if (endian() == LITTLE_ENDIAN)
        {
            const rank = buffer.readInt32LE();
            const shape = [  ];
            for (let i = 0; i < rank; ++i)
                shape.push(buffer.readInt32LE(4 * (i + 1)));
            const size = tf.util.sizeFromShape(shape);
            const array = [  ];
            for (let i = 0; i < size; ++i)
                array.push(buffer.readFloatLE(4 * (i + 1 + rank)));
            return tf.tensor3d(array, shape, "float32");
        }
        else if (endian() == BIG_ENDIAN)
        {
            const rank = buffer.readInt32BE();
            const shape = [  ];
            for (let i = 0; i < rank; ++i)
                shape.push(buffer.readInt32BE(4 * (i + 1)));
            const size = tf.util.sizeFromShape(shape);
            const array = [  ];
            for (let i = 0; i < size; ++i)
                array.push(buffer.readFloatBE(4 * (i + 1 + rank)));
            return tf.tensor3d(array, shape, "float32");
        }
        else throw new Error("unknown system endianness");
    }
    return tf.tidy("util.tnsToTensor", tidy);
}
