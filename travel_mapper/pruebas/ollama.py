from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
#
from langchain_google_genai import ChatGoogleGenerativeAI
#
from langchain.chat_models import ChatOpenAI
#from langchain_community import ChatOpenAI
#from langchain_openai import ChatOpenAI


# Load secrets API
def load_secrets():
    load_dotenv()  # necesario
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)

    open_ai_key = os.getenv("OPENAI_API_KEY")
    google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
    google_ai_key = os.getenv("GOOGLE_AI_API_KEY")

    return {
        "OPENAI_API_KEY": open_ai_key,
        "GOOGLE_MAPS_API_KEY": google_maps_key,
        "GOOGLE_AI_API_KEY": google_ai_key,
    }


# Cargamos las api_keys
secrets = load_secrets()
print(secrets)
openai_api_key = secrets["OPENAI_API_KEY"]
google_maps_key = secrets["GOOGLE_MAPS_API_KEY"]
google_ai_api_key = secrets["GOOGLE_AI_API_KEY"]

#
# Ollama example
#
# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
# MODEL_NAME = "llama2"
MODEL_NAME = "mistral"
# MODEL_NAME = "mixtral"

llm = ChatOllama(model=MODEL_NAME, temperature=0)
print('Ollama LLM Model: {}\n'.format(llm.model))

prompt = ChatPromptTemplate.from_template("Tell me a short story about {topic} in spanish language.")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# https://python.langchain.com/docs/expression_language/why
chain = prompt | llm | StrOutputParser()


# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print('Respuesta:')
print(chain.invoke({"topic": "Space travel"}))


#
# GoogleAI Gemini-pro example
#
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    google_api_key=google_ai_api_key,
)
print('ChatGoogleGenerativeAI LLM Model: {}\n'.format(llm.model))

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# https://python.langchain.com/docs/expression_language/why
chain = prompt | llm | StrOutputParser()


# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print('Respuesta:')
print(chain.invoke({"topic": "Space travel"}))


#
# OpenAI GPT example
#
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_api_key)
print('ChatOpenAI LLM Model: {}\n'.format(llm.model_name))

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# https://python.langchain.com/docs/expression_language/why
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production
print('Respuesta:')
print(chain.invoke({"topic": "Space travel"}))


###
# text_button.click(
#             travel_mapper.generate_without_leafmap,
#             inputs=[text_input_no_map, radio_no_map],
#             outputs=[text_output_no_map, query_validation_no_map],
#         )

### gemini-pro
#  * Day 1:
#   * Start your day at Plaza de Mayo (Av. Rivadavia 1625, C1024BAR, Buenos Aires), the iconic main square of Buenos Aires. Explore the nearby Casa Rosada (Piedras 1604, C1052BAR) and Metropolitan Cathedral (Calle Bolívar 923, C1024BAR).
#   * Have lunch at La Cabrera (Palermo Hollywood, Güemes 4781, C1414BAR), known for its authentic Argentine parrilla.
#   * In the afternoon, visit Recoleta Cemetery (Junín 1760, C1113BAR) and Palacio Barolo (Av. Costa Rica 5858, C1425BAR).
#   * End your day with a traditional Argentine tango show and dinner at El Querandí (Aréna 1361, C1425BAR).
# * Day 2:
#   * Begin your day at San Telmo Market (Plaza Dorrego, Defensa 970, S1078BAR) for a taste of local food and crafts.
#   * Take the Subte Line D to Palermo Viejo and explore its trendy streets, visit Palacio Paz (Av. Alvear 1625, C1425BAR), and have lunch at La Marola (Gorriti 5897, C1414BAR).
#   * In the afternoon, head to La Boca neighborhood (Caminito Solo de Mayo, La Boca) and visit the colorful houses and Camicie Rosse Museum.
#   * End your day with dinner at Don Julio (Guerico 4691, C1187BAR), a famous parrilla in La Boca.
# * Day 3:
#   * Start your day by visiting the modern neighborhood of Puerto Madero (Av. Pedro de Mendoza, Buenos Aires) and explore its waterfront promenade, Ecological Reserve, and Dock 4.
#   * Have lunch at La Pampa (Costanera Sur 6152, C1107BAR), a popular parrilla in Puerto Madero.
#   * In the afternoon, visit Teatro Colón (Cerrito 938, C1014BAR) and explore the historic neighborhood of San Nicolás (San Nicolás).
#   * End your day with dinner at El Solar de la Monja (Av. Caseros 5726, C1161BAR), a traditional Argentine restaurant in San Telmo.
# * Transit: Use Buenos Aires' extensive public transportation system, including buses and the Subte metro, to travel between locations. Walking is also recommended for exploring neighborhoods like Palermo Hollywood, La Boca, and San Telmo.

### google palm
#  * Day 1:
#   * Start your day at Plaza de Mayo (Av. Rivadavia 1625, C1024BAR, Buenos Aires), the iconic main square of Buenos Aires. Explore the nearby Casa Rosada (Piedras 1604, C1052BAR) and Metropolitan Cathedral (Calle Bolívar 923, C1024BAR).
#   * Have lunch at La Cabrera (Palermo Hollywood, Güemes 4781, C1414BAR), known for its authentic Argentine parrilla.
#   * In the afternoon, visit Recoleta Cemetery (Junín 1760, C1113BAR) and Palacio Barolo (Av. Costa Rica 5858, C1425BAR).
#   * End your day with a traditional Argentine tango show and dinner at El Querandí (Aréna 1361, C1425BAR).
# * Day 2:
#   * Begin your day at San Telmo Market (Plaza Dorrego, Defensa 970, S1078BAR) for a taste of local food and crafts.
#   * Take the Subte Line D to Palermo Viejo and explore its trendy streets, visit Palacio Paz (Av. Alvear 1625, C1425BAR), and have lunch at La Marola (Gorriti 5897, C1414BAR).
#   * In the afternoon, head to La Boca neighborhood (Caminito Solo de Mayo, La Boca) and visit the colorful houses and Camicie Rosse Museum.
#   * End your day with dinner at Don Julio (Guerico 4691, C1187BAR), a famous parrilla in La Boca.
# * Day 3:
#   * Start your day by visiting the modern neighborhood of Puerto Madero (Av. Pedro de Mendoza, Buenos Aires) and explore its waterfront promenade, Ecological Reserve, and Dock 4.
#   * Have lunch at La Pampa (Costanera Sur 6152, C1107BAR), a popular parrilla in Puerto Madero.
#   * In the afternoon, visit Teatro Colón (Cerrito 938, C1014BAR) and explore the historic neighborhood of San Nicolás (San Nicolás).
#   * End your day with dinner at El Solar de la Monja (Av. Caseros 5726, C1161BAR), a traditional Argentine restaurant in San Telmo.
# * Transit: Use Buenos Aires' extensive public transportation system, including buses and the Subte metro, to travel between locations. Walking is also recommended for exploring neighborhoods like Palermo Hollywood, La Boca, and San Telmo.

### ollama-mistral
#    * Día 1:
#   - Comenzar el día en Plaza de Mayo (Av. Rivadavia 628, C1052BAR Buenos Aires), visitar la Casa Rosada y la Catedral Metropolitana (Calle Bolívar 954, C1010BAR Buenos Aires).
#   - Después de la comida, tomar el Metro B a Palermo Soho (Avenida Serrano 2763, C1414BAR Buenos Aires) para disfrutar de la zona bohemia y visitar la Recoleta Cemetery (Junín 1760, C1114BAR Buenos Aires).
#   - Regresar a Monserrat (Calle Defensa 935, C1025BAR Buenos Aires) para cenar en un restaurante local.
#
# * Día 2:
#   - Empezar el día en La Boca (Camino Sarmiento s/n, 1425 Buenos Aires), visitar la Casa Rosada por la mañana y disfrutar de la zona colorida.
#   - Después de la comida, tomar el Colectivo 60 desde La Boca hasta Palermo Hollywood (Juan B. Justo 1425, C1415BAR Buenos Aires) para visitar el Teatro Colón (Cerrito 935, C1074BAR Buenos Aires).
#   - Regresar a Monserrat para cenar en otro restaurante local.
#
# * Día 3:
#   - Empezar el día en San Telmo (Defensa 862, C1069BAR Buenos Aires), visitar la Feria de San Telmo y disfrutar de los puestos artesanales.
#   - Después de la comida, tomar el Metro D desde San Telmo hasta Almagro (Avenida Caseros 2531, C1106BAR Buenos Aires) para visitar el Museo Evita (Calle Av. Caseros 4897, C1106BAR Buenos Aires).
#   - Regresar a Monserrat para cenar en un restaurante local y terminar la visita a Buenos Aires.
#
# Transit: A pie y transporte público (Metro B, Metro D y Colectivo 60)

### gtp-3.5
# - Día 1:
#   - Comienza tu día en el hotel ubicado en Av. Corrientes 1234, Buenos Aires.
#   - Toma el transporte público hasta la Plaza de Mayo, ubicada en Av. Rivadavia 100, Buenos Aires.
#   - Visita la Casa Rosada, ubicada en Balcarce 50, Buenos Aires.
#   - Camina hasta la Catedral Metropolitana, ubicada en Av. Rivadavia 100, Buenos Aires.
#   - Disfruta de un almuerzo en el restaurante "El Obrero", ubicado en Caffarena 64, Buenos Aires.
#   - Toma el transporte público hasta el barrio de San Telmo.
#   - Explora la Feria de San Telmo, ubicada en Defensa 100, Buenos Aires.
#   - Cena en el restaurante "La Brigada", ubicado en Estados Unidos 465, Buenos Aires.
#   - Regresa al hotel en transporte público.
#
# - Día 2:
#   - Comienza tu día en el hotel.
#   - Toma el transporte público hasta el barrio de La Boca.
#   - Visita el famoso Caminito, ubicado en Magallanes 800, Buenos Aires.
#   - Disfruta de un almuerzo en el restaurante "El Gran Paraiso", ubicado en Brandsen 923, Buenos Aires.
#   - Toma el transporte público hasta el barrio de Palermo.
#   - Explora los Bosques de Palermo, ubicados en Av. Sarmiento y Av. Figueroa Alcorta, Buenos Aires.
#   - Cena en el restaurante "La Cabrera", ubicado en Cabrera 5099, Buenos Aires.
#   - Regresa al hotel en transporte público.
#
# - Día 3:
#   - Comienza tu día en el hotel.
#   - Toma el transporte público hasta el barrio de Recoleta.
#   - Visita el famoso Cementerio de la Recoleta, ubicado en Junín 1760, Buenos Aires.
#   - Disfruta de un almuerzo en el restaurante "El Sanjuanino", ubicado en Posadas 1515, Buenos Aires.
#   - Toma el transporte público hasta el barrio de Puerto Madero.
#   - Pasea por la Reserva Ecológica Costanera Sur, ubicada en Av. Tristán Achával Rodríguez 1550, Buenos Aires.
#   - Cena en el restaurante "Siga La Vaca", ubicado en Alicia Moreau de Justo 1714, Buenos Aires.
#   - Regresa al hotel en transporte público.
#
# - Regresa al hotel en transporte público.

### gpt-4
# - Día 1:
#   - Comience su día en el Obelisco de Buenos Aires, Avenida 9 de Julio, Centro, Buenos Aires. Este es un monumento icónico de la ciudad.
#   - Camine hasta la Casa Rosada, Balcarce 50, C1064 CABA, Argentina. Esta es la oficina del presidente de Argentina y un edificio histórico.
#   - Almuerce en El Desnivel, Defensa 855, San Telmo, Buenos Aires. Este es un popular restaurante de parrilla argentina.
#   - Después del almuerzo, visite el Museo de Arte Latinoamericano de Buenos Aires (MALBA), Av. Pres. Figueroa Alcorta 3415, C1425 CLA, Buenos Aires.
#   - Cene en La Cabrera, Cabrera 5099, C1414 BGE, Buenos Aires. Este es un famoso restaurante de carne argentina.
#
# - Día 2:
#   - Comience su día en el Jardín Botánico Carlos Thays, Av. Santa Fe 3951, C1425 Buenos Aires.
#   - Camine hasta el Museo Evita, Lafinur 2988, C1425 FAB, Buenos Aires. Este museo está dedicado a la vida de Eva Perón.
#   - Almuerce en La Pecora Nera, Ayacucho 1785, C1112 AAB, Buenos Aires. Este es un restaurante italiano con un toque argentino.
#   - Después del almuerzo, tome el transporte público hasta el Cementerio de la Recoleta, Junín 1760, C1113 CABA, Buenos Aires. Este es un famoso cementerio donde están enterrados muchos personajes históricos argentinos.
#   - Cene en El Sanjuanino, Posadas 1515, C1112 CABA, Buenos Aires. Este es un restaurante que sirve comida regional argentina.
#
# - Día 3:
#   - Comience su día en el barrio de La Boca y visite el famoso Caminito, Dr. del Valle Iberlucea 1300, La Boca, Buenos Aires.
#   - Camine hasta el estadio de fútbol La Bombonera, Brandsen 805, C1161 CABA, Buenos Aires.
#   - Almuerce en El Obrero, Agustín R. Caffarena 64, C1157 CABA, Buenos Aires. Este es un restaurante tradicional de Buenos Aires.
#   - Después del almuerzo, tome el transporte público hasta el barrio de Palermo y visite el Parque 3 de Febrero, Av. Infanta Isabel 410, C1425 CABA, Buenos Aires.
#   - Cene en Don Julio, Guatemala 4691, C1425 Buenos Aires. Este es uno de los mejores restaurantes de parrilla de Buenos Aires.