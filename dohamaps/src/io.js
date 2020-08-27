
import * as fs from "fs";
import * as osPath from "path";

const fsAsync = fs.promises;

export async function loadFile(path)
{
    try { return await fsAsync.readFile(path); }
    catch (error) { console.log(error); }
}

export async function saveFile(path, name, buffer)
{
    try { return await fsAsync.writeFile(osPath.join(path, name), buffer); }
    catch (error) { console.log(error); }
}
