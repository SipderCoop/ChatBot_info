# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount

from API_INEGI import get_BIE_data


class MyBot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    async def on_message_activity(self, turn_context: TurnContext):
        user_message = turn_context.activity.text.lower()

        # Detectar si el usuario pregunta por la inflación
        if "inflación" in user_message or "inflacion" in user_message:
            # Llamar a la función get_BIE_data para obtener los datos de inflación
            # Asumiré que tenemos un indicator_id y last_data conocidos para el ejemplo
            indicator_id = 628208
            last_data = False
            json = get_BIE_data(indicator_id, last_data)
            inflation_data = json['serie']

            # Enviar los datos obtenidos al usuario
            message = f"La inflación en {inflation_data.index[0]} es de {inflation_data[inflation_data.index[0]]} "
            
        else:
            # Respuesta genérica si no se pregunta por la inflación
            message = f"Tú dijiste: '{turn_context.activity.text}'"
        
        await turn_context.send_activity(message)

    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome!")
