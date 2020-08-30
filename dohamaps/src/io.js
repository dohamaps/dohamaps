
import * as fs from "fs";
import * as osPath from "path";

const fsAsync = fs.promises;

async function exists(path)
{
    try
    {
        await fsAsync.access(path);
        return true;
    }
    catch (error)
    {
        return false;
    }
}

export async function loadFile(path)
{
    try { return await fsAsync.readFile(path); }
    catch (error) { console.error(error); }
}

export async function saveFile(path, name, buffer)
{
    try
    {
        if (!(await exists(path)))
            await fsAsync.mkdir(path, { recursive: true, mode: 0o777 });
        return await fsAsync.writeFile(osPath.join(path, name), buffer);
    }
    catch (error) { console.error(error); }
}
