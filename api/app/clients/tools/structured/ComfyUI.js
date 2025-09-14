// Generates image using ComfyUI webui's api (automatic1111)
const fs = require('fs');
const { z } = require('zod');
const path = require('path');
const axios = require('axios');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');
const { Tool } = require('@langchain/core/tools');
const { FileContext, ContentTypes } = require('librechat-data-provider');
const paths = require('~/config/paths');
const { logger } = require('~/config');
const { ComfyApi, CallWrapper, PromptBuilder, TSamplerName, TSchedulerName } = require("@saintno/comfyui-sdk");
const ExampleTxt2ImgWorkflowSD15 = require("./comfyui-example-woekflow-sd15.json");

const displayMessage =
  "ComfyUI displayed an image. All generated images are already plainly visible, so don't repeat the descriptions in detail. Do not list download links as they are available in the UI already. The user may download the images by clicking on them, but do not mention anything about downloading to the user.";

class ComfyUIAPI extends Tool {
  constructor(fields) {
    super();
    /** @type {string} User ID */
    this.userId = fields.userId;
    /** @type {ServerRequest | undefined} Express Request object, only provided by ToolService */
    this.req = fields.req;
    /** @type {boolean} Used to initialize the Tool without necessary variables. */
    this.override = fields.override ?? false;
    /** @type {boolean} Necessary for output to contain all image metadata. */
    this.returnMetadata = fields.returnMetadata ?? false;
    /** @type {boolean} */
    this.isAgent = fields.isAgent;
    if (fields.uploadImageBuffer) {
      /** @type {uploadImageBuffer} Necessary for output to contain all image metadata. */
      this.uploadImageBuffer = fields.uploadImageBuffer.bind(this);
    }

    this.name = 'comfyui';
    this.url = fields.COMFYUI_URL || this.getServerURL();
    this.description_for_model = `// Generate images and visuals using text.
// Guidelines:
// - ALWAYS use {{"prompt": "5+ detailed keywords", "negative_prompt": "5+ detailed keywords"}} structure for queries.
// - ALWAYS include the markdown url in your final response to show the user: ![caption](/images/id.png)
// - Visually describe the moods, details, structures, styles, and/or proportions of the image. Remember, the focus is on visual attributes.
// - Craft your input by "showing" and not "telling" the imagery. Think in terms of what you'd want to see in a photograph or a painting.
// - Here's an example for generating a realistic portrait photo of a man:
// "prompt":"photo of a man in black clothes, half body, high detailed skin, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
// "negative_prompt":"semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, out of frame, low quality, ugly, mutation, deformed"
// - Generate images only once per human query unless explicitly requested by the user`;
    this.description =
      "You can generate images using text with 'stable-diffusion'. This tool is exclusively for visual content.";
    this.schema = z.object({
      prompt: z
        .string()
        .describe(
          'Detailed keywords to describe the subject, using at least 7 keywords to accurately describe the image, separated by comma',
        ),
      negative_prompt: z
        .string()
        .describe(
          'Keywords we want to exclude from the final image, using at least 7 keywords to accurately describe the image, separated by comma',
        ),
    });
  }

  replaceNewLinesWithSpaces(inputString) {
    return inputString.replace(/\r\n|\r|\n/g, ' ');
  }

  getMarkdownImageUrl(imageName) {
    const imageUrl = path
      .join(this.relativePath, this.userId, imageName)
      .replace(/\\/g, '/')
      .replace('public/', '');
    return `![generated image](/${imageUrl})`;
  }

  returnValue(value) {
    if (this.isAgent === true && typeof value === 'string') {
      return [value, {}];
    } else if (this.isAgent === true && typeof value === 'object') {
      return [displayMessage, value];
    }

    return value;
  }

  getServerURL() {
    const url = process.env.COMFYUI_URL || '';
    if (!url && !this.override) {
      throw new Error('Missing COMFYUI_URL environment variable.');
    }
    return url;
  }

  async _call(data) {
    const url = this.url;
    const { prompt, negative_prompt, width, height } = data;
    const payload = {
      prompt,
      negative_prompt,
      cfg_scale: 4.5,
      steps: 6,
      width: width || 512,
      height: height || 512,
      seed: Math.floor(Math.random() * (999999999999 - 10000000000 + 1) + 10000000000)
    };

    // const api = new ComfyApi("http://localhost:8189").init();
    const api = new ComfyApi(url).init();

    const Txt2ImgPrompt = new PromptBuilder(
      ExampleTxt2ImgWorkflowSD15,
      ["positive", "negative", "checkpoint", "seed", "batch", "step", "cfg", "sampler", "sheduler", "width", "height"],
      ["images"]
    )
      .setInputNode("checkpoint", "4.inputs.ckpt_name")
      .setInputNode("seed", "3.inputs.seed")
      .setInputNode("batch", "5.inputs.batch_size")
      .setInputNode("negative", "7.inputs.text")
      .setInputNode("positive", "6.inputs.text")
      .setInputNode("cfg", "3.inputs.cfg")
      .setInputNode("sampler", "3.inputs.sampler_name")
      .setInputNode("sheduler", "3.inputs.scheduler")
      .setInputNode("step", "3.inputs.steps")
      .setInputNode("width", "5.inputs.width")
      .setInputNode("height", "5.inputs.height")
      .setOutputNode("images", "9");

    const workflow = Txt2ImgPrompt.input(
      "checkpoint",
      // "SDXL/realvisxlV40_v40LightningBakedvae.safetensors",
      "sd-v1-5-pruned-emaonly-fp16.safetensors",
      /** Use the client's osType to encode the path */
      api.osType
    )
      .input("seed", payload.seed)
      .input("step", 6)
      .input("cfg", 1)
      // .input<TSamplerName>("sampler", "dpmpp_2m_sde_gpu")
      // .input<TSchedulerName>("sheduler", "sgm_uniform")
      .input("width", 512)
      .input("height", 512)
      .input("batch", 1)
      .input("positive", prompt)
      .input("negative", negative_prompt);

    async function requestTextToImage() {
      return new Promise((resolve, reject) => {
        new CallWrapper(api, workflow)
          .onFinished(async (data) => {
            if (Array.isArray(data.images?.images) && data.images.images.length > 0) {
              // imgUrl value like: http://localhost:8000/view?filename=sd15lcm_00002_.png&type=output&subfolder=lawn
              const imgUrl = api.getPathImage(data.images.images[0])
              console.log(imgUrl)

              try {
                const imgBase64 = await getImageBase64(imgUrl);
                // console.log('Base64 image data:', imgBase64.substring(0, 100) + '...');
                resolve(imgBase64.slice(22));
              } catch (error) {
                console.error('Failed to convert image to base64:', error);
                reject(error);

              }

            }
          })
          .onFailed((error) => {
            console.error('Failed to generate image:', error);
            reject(error);
          })
          .onProgress((status) => console.log("image generating...", status.node, `${status.value}/${status.max}`))
          .run();
      });

    }

    let generationResponse;
    try {
      // generationResponse = await axios.post(`${url}/sdapi/v1/txt2img`, payload);
      generationResponse = await requestTextToImage();
    } catch (error) {
      logger.error('[ComfyUI] Error while generating image:', error);
      return 'Error making API request.';
    }
    const image = generationResponse;
    // console.log('cfy- Image data type:', typeof image, this.isAgent);

    /** @type {{ height: number, width: number, seed: number, infotexts: string[] }} */
    let info = {};
    try {
      info = JSON.parse(generationResponse?.data?.info);
    } catch (error) {
      logger.warn('[ComfyUI] Error while getting image metadata:', error);
    }
    info = { ...payload, ...info }

    const file_id = uuidv4();
    const imageName = `${file_id}.png`;
    const { imageOutput: imageOutputPath, clientPath } = paths;
    const filepath = path.join(imageOutputPath, this.userId, imageName);
    this.relativePath = path.relative(clientPath, imageOutputPath);

    if (!fs.existsSync(path.join(imageOutputPath, this.userId))) {
      fs.mkdirSync(path.join(imageOutputPath, this.userId), { recursive: true });
    }

    try {
      if (this.isAgent) {
        const content = [
          {
            type: ContentTypes.IMAGE_URL,
            image_url: {
              url: `data:image/png;base64,${image}`,
            },
          },
        ];

        const response = [
          {
            type: ContentTypes.TEXT,
            text: displayMessage,
          },
        ];
        // console.log(';; cfy- return')
        return [response, { content }];
      }

      // ❓ it seems unused below
      const buffer = Buffer.from(image.split(',', 1)[0], 'base64');
      if (this.returnMetadata && this.uploadImageBuffer && this.req) {
        const file = await this.uploadImageBuffer({
          req: this.req,
          context: FileContext.image_generation,
          resize: false,
          metadata: {
            buffer,
            height: info.height,
            width: info.width,
            bytes: Buffer.byteLength(buffer),
            filename: imageName,
            type: 'image/png',
            file_id,
          },
        });

        const generationInfo = info?.infotexts[0].split('\n').pop();
        return {
          ...file,
          prompt,
          metadata: {
            negative_prompt,
            seed: info.seed,
            info: generationInfo,
          },
        };
      }

      await sharp(buffer)
        .withMetadata({
          iptcpng: {
            parameters: info?.infotexts[0],
          },
        })
        .toFile(filepath);
      this.result = this.getMarkdownImageUrl(imageName);
    } catch (error) {
      logger.error('[ComfyUI] Error while saving the image:', error);
    }

    return this.returnValue(this.result);
  }
}

module.exports = ComfyUIAPI;



/**
* Fetches an image from a URL and converts it to base64 string
* Works in both Node.js and browser environments
*/
async function getImageBase64(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch comfyui image: ${response.status} ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const resMimeType = response.headers.get('content-type')
    const mimeType = resMimeType || 'image/png';

    // Check if we're in Node.js environment
    if (typeof Buffer !== 'undefined') {
      // Node.js environment
      const buffer = Buffer.from(arrayBuffer);
      const base64 = buffer.toString('base64');
      return `data:${mimeType};base64,${base64}`;
    } else {
      // Browser environment
      return new promises((resolve, reject) => {
        const blob = new Blob([arrayBuffer], { type: mimeType });
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }
  } catch (error) {
    console.error('Error converting image to base64:', error);
    throw error;
  }
}
