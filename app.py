import os
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()


# Configuração da API da Groq
api_key = os.getenv("GROQ_API_KEY")
client = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")

# Configuração da memória
MEMORIA = ConversationBufferWindowMemory(k=5)  # Mantém as últimas 5 mensagens

# Carregamento do arquivo CSV previamente carregado
CAMINHO_CSV = "dados.csv"  # Substitua pelo caminho do seu arquivo CSV
def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

dados_documento = carrega_csv(CAMINHO_CSV)

# Configuração do sistema
system_message = f"""Você é um assistente inteligente chamado Victor.
Você possui conhecimento avançado em processos de reciclagem, gestão de resíduos, sustentabilidade, e pode responder com base nos seguintes dados:

### 
{dados_documento}
###

Utilize as informações acima para embasar suas respostas. 
Se a informação for insuficiente, peça mais contexto ou informe que os dados não estão disponíveis.
Sempre que houver $ na sua saída, substitua por S.
A coluna "valor_das" representa Documento de Arrecadação do Simples Nacional pago mensalmente.
A coluna "data_pagamento" refere-se à data que a empresa pagou o das.

A empresa com a qual você está conversando é a Nova Era Reciclagem. 
Os sócios da empresa são Matheus Henrique e Márcia. 
Sempre pergunte com quem está falando para saber com qual sócio você está convesando.
Eles são especializados na gestão e reciclagem de resíduos, buscando sempre a sustentabilidade e a inovação no setor de reciclagem de materiais. Seu objetivo é otimizar o processo de reciclagem e contribuir para um ambiente mais sustentável.

Você pode fornecer informações sobre a empresa, suas práticas, dados financeiros, ou qualquer outro aspecto relacionado à reciclagem, sustentabilidade e questões financeiras que envolvam a empresa Nova Era Reciclagem. 
Se precisar de mais informações, esteja à vontade para perguntar.
"""


template = ChatPromptTemplate.from_messages([
    ('system', system_message),
    ('placeholder', '{chat_history}'),
    ('user', '{input}')
])

chain = template | client

# Página de chat
def pagina_chat():
    st.header('🤖 Analista Contábil', anchor=None)  # Corrigido

    memoria = st.session_state.get('memoria', MEMORIA)

    for mensagem in memoria.buffer_as_messages:
        chat_display = st.chat_message(mensagem.type)
        chat_display.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o Analista')
    if input_usuario:
        # Exibir mensagem do usuário
        chat_display = st.chat_message('human')
        chat_display.markdown(input_usuario)

        # Resposta do modelo
        chat_display = st.chat_message('ai')
        resposta = chat_display.write_stream(chain.stream({
            'input': input_usuario, 'chat_history': memoria.buffer_as_messages
        }))
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)

        st.session_state['memoria'] = memoria

def sidebar():
    # Adicionar a logo no topo da barra lateral
    st.image("logo.png", use_container_width=True)  # Substitua pelo caminho da sua logo

    tabs = st.tabs(['Conversas', 'Configurações'])
    with tabs[0]:
         if st.button('Apagar Histórico de Conversas'):
            st.session_state['memoria'] = MEMORIA
    
    with tabs[1]:
        st.title("Configurações")
        st.write("Este projeto utiliza o modelo Llama 3.3.")


# Interface principal
def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()