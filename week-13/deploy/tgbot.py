import os

from aiotg import Bot, Chat
import emoji
import imageio
from services import ClassifyModel

model = ClassifyModel()

bot = Bot(api_token=os.getenv("TG_TOKEN"))


@bot.command("/start")
async def start(chat: Chat, match):
    return chat.reply("Send me photo of cat or dog.")


async def process_image(binary_data, chat_to_reply: Chat):
    # Convert binary data to numpy.ndarray image
    image = imageio.imread(binary_data)

    # Do the magic
    tag = await model.predict.call(image)
    e = emoji.emojize(f":{tag}:")

    # Simple text response
    await chat_to_reply.reply(f"I think this is {tag}: {e}!")


@bot.handle("photo")
async def handle_photo(chat: Chat, photos):
    # Get image binary data
    meta = await bot.get_file(photos[-1]["file_id"])
    resp = await bot.download_file(meta["file_path"])
    data = await resp.read()

    await process_image(data, chat)


@bot.handle("document")
async def handle_document(chat: Chat, document):
    # Get image binary data
    meta = await bot.get_file(document["file_id"])
    resp = await bot.download_file(meta["file_path"])
    data = await resp.read()

    await process_image(data, chat)


bot.run()
