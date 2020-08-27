
const tf = require("@tensorflow/tfjs-node");
const dohamaps = require("../");

async function tnsTest()
{
    try
    {
        var buffer = await dohamaps.io.loadFile("/Users/admin/Desktop/frong.png");
        var tensor = dohamaps.image.imageToTensor(buffer);
        tensor = tf.concat([ tensor.clone(), tensor.clone(), tensor.clone() ], 2);
        tensor = dohamaps.image.cropRandom(tensor, 64, 64);
        buffer = await dohamaps.util.tensorToTns(tensor);
        await dohamaps.io.saveFile("/Users/admin/Desktop/", "fring.tns", buffer);
    }
    catch (error) { console.log(error); }
}
async function imageTest()
{
    try
    {
        var buffer = await dohamaps.io.loadFile("/Users/admin/Desktop/fring.tns");
        var tensor = dohamaps.util.tnsToTensor(buffer);
        buffer = await dohamaps.image.tensorToImage(tensor, 4);
        await dohamaps.io.saveFile("/Users/admin/Desktop/", "fring.png", buffer);
    }
    catch (error) { console.log(error); }
}

async function test()
{
    try
    {
        await tnsTest();
        await imageTest();
    }
    catch (error) { console.log(error); }
}

test();
