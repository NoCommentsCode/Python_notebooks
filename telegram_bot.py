import wolframalpha as wa
import telebot

app_id = 'ELQ8HA-KQKRHE2KVW'
client = wa.Client(app_id)
def calculate(question):
    try:
        res = client.query(question)
        list = next(res.results)['subpod']
        answer = ''
        try:
            answer = list['img']['@alt']
        except:
            try:
                for i in list:
                    answer += i['img']['@alt'] + '\n'
            except:
                answer = 'Нихуя не получилось'
    except:
        answer = 'Че то не то'
    return answer

token = '5367239215:AAGTsWVilSMwJTlCBu-Ai_Iga9GjwpgsBxI'
#print(str(wiki.summary('euler equation')))
bot = telebot.TeleBot(token = token)
# Функция, обрабатывающая команду /start
@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Что считаем?')
# Получение сообщений от юзера
@bot.message_handler(content_types=["text"])
def handle_text(message):
    bot.send_message(message.chat.id, calculate(message.text))
# Запускаем бота
bot.polling(none_stop=True, interval=0)