
const http = require('https');
const fs = require('fs');

// ms between gif downloads
// Increase this if you encounter errors while downloading!
// 25ms should be stable on gigabit. It's about 450Mbps
const DOWNLOAD_INTERVAL = 25;

const gifs = fs
    .readFileSync('./data/gif-url-captions.tsv')
    .toString()
    .split('\n')
    .map(line => line.split('\t'))
    .map(lines => ({
        url: lines[0],
        caption: (lines[1] ? lines[1] : '')
    }));
gifs.pop();

const gifIds = gifs
    .reduce((urls, gif) => (urls[gif.url] === undefined ? urls[gif.url]={uid:urls.__i++} : '')!=-1 && urls, {__i:0})
delete gifIds.__i;

const gifCaptionIdMap = gifs.map(gif => gifIds[gif.url].uid + '\t' + gif.caption).join('\n');
fs.writeFileSync('./captions.txt', gifCaptionIdMap);

const gifUrls = [gifIds]
    .map(obj => Object.keys(obj).filter(key => obj.hasOwnProperty(key)))[0]
    .reduce((urls, url) => urls.push([url, gifIds[url].uid]) && urls, [])

// Broken GIFs:
// [ 57709, 68784, 70375, 74522, 83553, 83696, 87582, 87877, 88401 ]
// console.log(gifUrls.filter(u => u[0].indexOf('42.media') > 0).map(u => u[1]));

let index = 0;
setInterval(() => {
    const gif = gifUrls[index++];
    if (!gif || gif[0].indexOf('42.media') > 0) return;
    const name = `./gifs/${gif[1]}.gif`;
    console.log(gif[0], name);
    try {
        http.get(
            gif[0], 
            response => {
                const f = fs.createWriteStream(name);
                response.pipe(f);
                f.on('finish', () => f.close());
            }, 
            err => console.log(err)
        )
    }
    catch (e) {
        console.log(e);
    }
}, DOWNLOAD_INTERVAL);