# Out of the scope:
# TGLANG_LANGUAGE_FUNC
# TGLANG_LANGUAGE_TL

import requests
import json
import tqdm
import re
import os
import random

import sys

sys.path.append(".")
import utils.lang_constructs


def extract_answer(text):
    lines = text.splitlines()
    if "sure" in lines[0].lower():
        lines = lines[1:]

    for i, line in enumerate(lines):
        if "Translation:" in line:
            lines = lines[:i]
            break

    if len(lines) == 0:
        return None

    return "\n".join(lines)


def ask_llama(prompt, context=None):
    url = "http://localhost:11434/api/generate"

    req = {"model": "llama2", "stream": False, "prompt": prompt}
    if context is not None:
        req["context"] = context

    while True:
        try:
            response = requests.request(
                "POST", url, data=json.dumps(req), timeout=60
            ).json()
            break
        except:
            continue

    return response["response"], response["context"]


def get_code(
    limit=10000,
    reset_context=4,
    save_root=None,
):
    all_requests = []
    for i in range(limit):
        request = [gen_prompt()]
        all_requests.append(request)

    code_blocks = []

    if save_root is not None:
        save_dir = os.path.join(save_root, "OTHER")
        os.makedirs(save_dir, exist_ok=True)

    for i, request in enumerate(tqdm.tqdm(all_requests, desc="Processing OTHER")):
        if save_root is not None:
            request_save_dir = os.path.join(save_dir, str(i))
            if os.path.exists(request_save_dir) and len(
                os.listdir(request_save_dir)
            ) == len(request):
                continue

            os.makedirs(request_save_dir, exist_ok=True)

        context = None
        for j, prompt in enumerate(request):
            if j % reset_context == 0:
                context = None
            response, curr_context = ask_llama(prompt, context)
            context = curr_context

            code = extract_answer(response)
            if code is None:
                continue
            code_blocks.append(code)

            if save_root is not None:
                with open(os.path.join(request_save_dir, "%d.txt" % j), "w") as f:
                    f.write(code)

    return code_blocks


def gen_prompt():
    task = random.choice(tasks)
    character = random.choice(characters)
    country, lang = random.choice(langs)

    symbols = True if random.random() < 0.1 else False
    emojis = True if random.random() < 0.1 else False
    structures = True if random.random() < 0.1 else False
    tables = True if random.random() < 0.05 else False
    slang = True if random.random() < 0.1 else False
    max_len = random.randint(1, 7)

    use_structures = ""
    if structures:
        use_structures = "Use some structure in your text, for example add numeration or points or any other."
    use_symbols = ""
    if symbols:
        use_symbols = "Use special symbols in your text, for example highlight some words or define titles."
    use_emojis = ""
    if emojis:
        use_emojis = "Use emojis in your text!"
    add_tables = ""
    if tables:
        add_tables = "Add some tables to your text!"
    use_slang = ""
    if slang:
        use_slang = "Use slang in your text!"

    prompt = f"""You are {character} from {country}. You need to {task}. {use_symbols} {use_emojis} {use_structures} {add_tables} {use_slang}
Not necessary to make it correct or true, but make sure that it looks very natural.
WRITE YOUR TEXT IN {lang.upper()} LANGUAGE. USE NO MORE THAN {max_len} SENTENCES OR LINES.
DON'T COMMENT YOUR TEXT. JUST AN ANSWER.
So, you are {character}, please begin:"""

    return prompt


characters = [
    "a curious scientist",
    "an avid gardener",
    "a passionate chef",
    "a freelance writer",
    "a diligent student",
    "an experienced pilot",
    "a creative artist",
    "a tech enthusiast",
    "a seasoned fisherman",
    "a professional gamer",
    "a dedicated nurse",
    "an aspiring musician",
    "a skilled carpenter",
    "a strategic planner",
    "a fashion blogger",
    "a fitness coach",
    "a vintage collector",
    "a travel vlogger",
    "a crafty knitter",
    "an amateur astronomer",
    "a cryptocurrency trader",
    "an animal rescuer",
    "a mystery novelist",
    "a coffee connoisseur",
    "a historical reenactor",
    "a robotics engineer",
    "a social media influencer",
    "a thrill-seeking adventurer",
    "a professional clown",
    "a special effects makeup artist",
    "a competitive chess player",
    "a local politician",
    "a wedding planner",
    "an urban explorer",
    "a paranormal investigator",
    "a reality TV star",
    "a voiceover artist",
    "a theme park designer",
    "a professional critic",
    "a drone pilot",
    "a digital nomad",
    "a sommelier",
    "a genealogist",
    "an escape room designer",
    "a ghostwriter",
    "a puppeteer",
    "a graffiti artist",
    "a surf instructor",
    "a bounty hunter",
    "a stunt double",
    "a sand sculptor",
    "a hot air balloon pilot",
    "an ice sculptor",
    "a chocolatier",
    "a fire performer",
    "a sword swallower",
    "an antiques dealer",
    "a tarot card reader",
    "a fragrance designer",
    "a professional sleeper",
    "a tech news blogger",
    "a cybersecurity expert",
    "an SEO specialist",
    "a digital marketer",
    "a web developer",
    "a UX/UI designer",
    "an online tutor",
    "a data analyst",
    "a cloud computing consultant",
    "an AI researcher",
    "a video game streamer",
    "an e-commerce entrepreneur",
    "a virtual event organizer",
    "a podcast host",
    "a mobile app developer",
    "a network engineer",
    "a blockchain developer",
    "a meme creator",
    "an email marketing manager",
    "a social media manager",
    "an affiliate marketer",
    "a virtual assistant",
    "a domain flipper",
    "a tech support specialist",
    "an online community moderator",
    "a content strategist",
    "an internet law attorney",
    "a database administrator",
    "a machine learning engineer",
    "a phishing scammer",
    "an online surveyor",
    "a captcha solver",
    "an adware promoter",
    "a spam email generator",
    "a VPN service reviewer",
    "a web scraping expert",
    "an online privacy advocate",
    "a digital rights activist",
    "a freelance graphic designer",
    "a YouTube channel manager",
    "an internet historian",
    "a crowdfunding campaigner",
    "a digital nomad blogger",
    "an online influencer",
    "a cryptocurrency blogger",
    "a dark web explorer",
    "a virtual reality content creator",
    "an online dating consultant",
    "a fake review writer",
    "a website flipper",
    "an internet hoax debunker",
    "an online scam investigator",
    "a social media influencer",
    "a digital art seller",
    "a tech gadget reviewer",
    "an online petition creator",
    "a Twitch moderator",
    "a viral content creator",
    "an emoji designer",
    "a web accessibility consultant",
    "a food review blogger",
    "a travel experience blogger",
    "a personal finance advisor blogger",
    "a DIY crafts blogger",
    "a parenting tips blogger",
    "a sustainable living blogger",
    "a luxury fashion blogger",
    "a minimalist lifestyle blogger",
    "a home renovation blogger",
    "a personal growth blogger",
    "a street style photographer blogger",
    "a digital nomad lifestyle blogger",
    "a yoga and wellness blogger",
    "a budget travel blogger",
    "a beauty and skincare blogger",
    "a vegan cooking blogger",
    "a professional development blogger",
    "a film critique blogger",
    "a book review blogger",
    "a gardening advice blogger",
    "a sports analysis blogger",
    "a wildlife photography blogger",
    "a cultural analysis blogger",
    "a historical events blogger",
    "a tech gadget review blogger",
    "a comedy and satire blogger",
    "a science communication blogger",
    "a music industry trends blogger",
    "a car review and news blogger",
    "a wedding planning blogger",
    "a baby products reviewer blogger",
    "a pet care and advice blogger",
    "a celebrity gossip blogger",
    "a local news and events blogger",
    "a mental health awareness blogger",
    "a language learning blogger",
    "a coffee culture blogger",
    "a graphic design trends blogger",
    "a street art and murals blogger",
    "a video gaming blogger",
    "a sustainable fashion blogger",
    "a home cooking and recipes blogger",
    "a personal trainer and fitness blogger",
    "a life in a foreign country blogger",
    "a home automation and tech blogger",
    "a budgeting and saving money blogger",
    "a kids' activities and crafts blogger",
    "a digital marketing strategies blogger",
    "a microbrewery and craft beer blogger",
    "a local restaurant review blogger",
    "a hiking and outdoor adventure blogger",
    "a astrology and horoscope blogger",
    "a college life and study tips blogger",
    "a mobile technology trends blogger",
    "a career coaching and advice blogger",
    "a short story and fiction blogger",
    "a urban exploration and travel blogger",
    "a self-sustaining homestead blogger",
    "a high-end audio equipment blogger",
    "a vintage clothing and style blogger",
    "a new parent",
    "a college student",
    "a retired veteran",
    "a small business owner",
    "a high school teacher",
    "a professional athlete",
    "an amateur gardener",
    "a local community leader",
    "a freelance photographer",
    "a nursing student",
    "a car enthusiast",
    "a travel blogger",
    "a passionate environmentalist",
    "a seasoned chef",
    "a tech startup founder",
    "a fashion design student",
    "a volunteer firefighter",
    "a city council member",
    "a long-distance runner",
    "a professional musician",
    "a civil rights activist",
    "a local librarian",
    "a professional comedian",
    "a public health researcher",
    "a real estate agent",
    "an art history major",
    "a full-time RVer",
    "a yoga instructor",
    "a wildlife conservationist",
    "a second-grade teacher",
    "a dance choreographer",
    "a solar energy consultant",
    "a comic book collector",
    "a film production assistant",
    "a boutique owner",
    "a mountain guide",
    "an urban planner",
    "a biomedical engineer",
    "a professional gamer",
    "a mixologist",
    "a beekeeper",
    "a drone hobbyist",
    "a political campaign manager",
    "a jazz musician",
    "a pastry chef",
    "a genetic counselor",
    "a ceramics artist",
    "a forensic analyst",
    "an antique restorer",
    "a landscape architect",
    "a marine biologist",
    "a quantum computing researcher",
    "a professional dog trainer",
    "a mural artist",
    "a classical pianist",
    "a neurosurgeon",
    "a world traveler",
    "a barista",
    "an early childhood educator",
    "a virtual reality developer",
    "a professional astrologer",
]

langs = [
    ["China", "Mandarin"],
    ["India", "Hindi"],
    ["United States", "English"],
    ["United States", "English"],
    ["United States", "English"],
    ["United States", "English"],
    ["United States", "English"],
    ["United States", "English"],
    ["United States", "English"],
    ["Pakistan", "Urdu"],
    ["Brazil", "Portuguese"],
    ["Nigeria", "Hausa"],
    ["Bangladesh", "Bengali"],
    ["Russia", "Russian"],
    ["Russia", "Russian"],
    ["Russia", "Russian"],
    ["Russia", "Russian"],
    ["Mexico", "Spanish"],
    ["Japan", "Japanese"],
    ["Ethiopia", "Amharic"],
    ["Egypt", "Arabic"],
]

tasks = [
    "write a brief product review",
    "share a recent travel experience",
    "post a daily fitness tip",
    "update your followers about a tech gadget",
    "compose a motivational quote",
    "give a quick language learning tip",
    "discuss a current news event",
    "offer a home DIY hack",
    "share a favorite recipe",
    "post an interesting fact about your city",
    "give an opinion on a trending topic",
    "ask a thought-provoking question",
    "share a personal achievement",
    "give advice on a common problem",
    "post a joke or a funny observation",
    "explain a complex idea simply",
    "share a helpful programming tip",
    "post a career development insight",
    "discuss a recent scientific discovery",
    "offer a tip on sustainable living",
    "share a photography tip",
    "give a quick movie or book review",
    "share a health and wellness tip",
    "post an inspirational story",
    "give tips on effective remote work",
    "share insights on the latest software update",
    "offer a gardening tip",
    "give a brief political analysis",
    "post about a new music album",
    "share your favorite exercise routine",
    "give advice on balancing work and life",
    "post a quick cooking tip",
    "discuss a historical event",
    "offer insights on a financial market trend",
    "give a pet care tip",
    "share a fun fact about technology",
    "post a brief travel guide",
    "give a beauty or skincare tip",
    "share an art or craft idea",
    "discuss a recent sports event",
    "offer a parenting tip",
    "give a quick business strategy",
    "post a coding challenge solution",
    "share a DIY fashion hack",
    "give insights on the latest mobile apps",
    "post a short story or poem",
    "share a cultural insight",
    "give a brief review of a local event",
    "post a teaser of your upcoming project",
    "share a personal development tip",
    "offer a quick tip on graphic design",
    "post about a recent gaming experience",
    "share a unique coffee recipe",
    "give a quick lesson in photography",
    "post your thoughts on environmental conservation",
    "share a study tip for students",
    "offer advice on buying tech gadgets",
    "give a brief analysis of a legal topic",
    "post an update on a personal project",
    "share a tip on creating engaging content",
    "critique a specific web design trend",
    "explain a recent algorithm update in a search engine",
    "debate the merits of a new programming language",
    "review a specific model of smartphone",
    "discuss the impact of a recent data privacy regulation",
    "analyze a particular trend in digital marketing",
    "compare two competing software tools",
    "offer a detailed tutorial on a complex photo editing technique",
    "explain the steps for setting up a home automation system",
    "give a comprehensive guide on starting a podcast",
    "discuss the ethical implications of AI in healthcare",
    "review the latest updates in a specific video game",
    "offer a step-by-step guide to building a PC",
    "analyze a recent cybersecurity breach and its implications",
    "explain the benefits of a specific coding framework",
    "provide a detailed analysis of a recent tech IPO",
    "give insights on optimizing website SEO for a niche market",
    "discuss the impact of social media algorithms on news distribution",
    "provide a comprehensive review of a new camera release",
    "explain the process of developing a mobile app for iOS",
    "offer detailed advice on cloud storage solutions for small businesses",
    "analyze a specific trend in e-commerce customer behavior",
    "provide a step-by-step guide on creating an effective online ad campaign",
    "discuss the latest developments in renewable energy technologies",
    "offer a detailed breakdown of a specific cryptocurrency's performance",
    "provide insights on implementing machine learning in financial forecasting",
    "give a step-by-step tutorial on advanced Excel functions",
    "discuss strategies for securing IoT devices in smart homes",
    "provide an in-depth review of a specific model of electric car",
    "explain the intricacies of GDPR compliance for online businesses",
    "offer a guide on creating virtual reality content",
    "discuss the latest advancements in 3D printing technology",
    "provide a detailed analysis of a major tech company's annual report",
    "offer a tutorial on setting up a professional live streaming setup",
    "analyze the latest UI/UX design trends for mobile apps",
    "provide a comprehensive guide on affiliate marketing for bloggers",
    "discuss the role of blockchain in supply chain management",
    "give a detailed comparison of various project management software",
    "provide a step-by-step guide to creating a YouTube channel",
    "analyze the impact of augmented reality in retail",
    "offer insights into managing online community engagement",
    "discuss the latest trends in wearable tech",
    "provide a detailed tutorial on advanced Python programming",
    "give a comprehensive review of a major online learning platform",
    "offer a step-by-step guide to effective email marketing strategies",
    "analyze the role of influencers in shaping consumer behavior",
    "provide insights on optimizing user experience in web design",
    "discuss the latest trends in digital art and NFTs",
    "give a detailed breakdown of voice search optimization techniques",
    "offer a guide to implementing sustainable practices in digital businesses",
    "discuss the latest advancements in drone technology",
    "provide a comprehensive analysis of a recent major tech merger",
    "give insights on enhancing cybersecurity in small enterprises",
    "offer a tutorial on creating engaging content for social media",
    "analyze the impact of 5G technology on mobile communications",
    "share your morning routine",
    "post a picture of your breakfast",
    "write about your workout plan",
    "give a review of a book you're reading",
    "share a work-from-home tip",
    "post your favorite coffee recipe",
    "discuss your latest DIY project",
    "share a pet care tip",
    "write about a local event you attended",
    "post your evening skincare routine",
    "give a review of a new TV show",
    "share a quick healthy snack idea",
    "write about managing work-life balance",
    "post a quote that inspires you",
    "share a budgeting or saving tip",
    "write about a recent hike or nature trip",
    "post your thoughts on a current event",
    "share a beginner's guide to meditation",
    "write about an interesting podcast episode",
    "post a favorite childhood memory",
    "share tips for staying organized",
    "write about a recent cooking fail or success",
    "post a picture of your workspace",
    "share how you relax after a long day",
    "write about a goal you recently achieved",
    "post a simple gardening hack",
    "share your experience with a new hobby",
    "write about overcoming a challenge",
    "post a tutorial on a craft you made",
    "share a tip for learning a new language",
    "write about a local delicacy",
    "post a fun weekend activity idea",
    "share your thoughts on a new music album",
    "write about a memorable travel experience",
    "post a creative writing piece",
    "share a tip for sustainable living",
    "write about a cultural festival",
    "post a picture of your favorite outfit",
    "share your experience with a new exercise",
    "write about a funny incident",
    "post a recipe for a family dinner",
    "share tips for managing stress",
    "write about a recent movie you watched",
    "post an interesting historical fact",
    "share your favorite photography spots",
    "write about a life lesson you learned",
    "post a review of a local restaurant",
    "share a story about your pet",
    "write about a day in your life",
    "post your favorite motivational speech",
    "share a unique ice cream flavor you tried",
    "write about a recent art exhibit",
    "post a tip for first-time travelers",
    "share a review of a tech gadget you use",
    "write about an unusual hobby",
    "post a childhood recipe",
    "share a tip for improving sleep",
    "write about a random act of kindness",
    "post a picture of a sunset or sunrise",
    "share your experience with a new app",
    "write about a trend you're following",
    "post a guide to a local market",
    "share a story about your best friend",
    "write about a recent concert or show",
    "post a tip for staying hydrated",
    "share a home decor idea",
    "write about your favorite board game",
    "post a tip for dealing with jet lag",
    "share your thoughts on a philosophical question",
    "write about a family tradition",
    "post a guide to a weekend getaway",
    "share your method for making a tough decision",
    "write about an interesting local legend",
    "post a homemade gift idea",
    "share a childhood hobby you revisited",
    "write about an inspiring person in your life",
    "post your favorite dessert recipe",
    "share tips for reducing screen time",
    "write about a personal growth journey",
    "post a fun fact about your city",
    "share your experience at a farmer's market",
    "write about a song that changed your life",
    "post a guide to a relaxing evening",
    "share a memory that makes you smile",
    "write about a new cultural experience",
    "post a tip for improving focus",
    "share a review of a new restaurant",
    "write about a small victory",
    "post a picture of nature in your area",
    "share a technique for creative brainstorming",
    "write about a surprising fact you learned",
    "post a guide to a day trip in your city",
    "share a favorite childhood game",
    "write about a recent purchase you loved",
    "post a tip for dealing with a bad day",
    "share a story about a recent adventure",
    "write about your favorite seasonal activity",
    "post a review of a local coffee shop",
    "share your experience with a new fitness trend",
    "write about a meaningful conversation",
]

if __name__ == "__main__":
    res = get_code(limit=10000, save_root="../datasets/llama/stories/")
