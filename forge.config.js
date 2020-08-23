module.exports =
{
    github_repository:
    {
        name: "dohamaps",
        owner: "dohamaps"
    },
    packagerConfig: { icon: "./electron/icon.icns" },
    publishers:
    [
        {
            name: "@electron-forge/publisher-github",
            config:
            {
                repository:
                {
                    name: "dohamaps",
                    owner: "dohamaps"
                },
                prerelease: false,
                draft: false
            }
        }
    ],
    makers:
    [
        {
            name: "@electron-forge/maker-squirrel",
            config: { name: "dohamaps" }
        },
        {
            name: "@electron-forge/maker-zip",
            platforms: [ "darwin" ]
        },
        {
            name: "@electron-forge/maker-deb",
            config: {  }
        },
        {
            name: "@electron-forge/maker-rpm",
            config: {  }
        }
    ]
}
