import os
import pandas as pd


vignette_file = open("vignette_bank.txt", "r")
data = vignette_file.read()
vignette_list = data.split('\n')
counter = 1
x = 0

while x < len(vignette_list):
    dialogue = { 'vignette': vignette_list[x]} 
# start of the dictionary 

    the_end = False
    counter_turn = 0


    print("patient1")
    vals = dialogue.values()
    dialogue_current = "\n".join(vals)
    if counter_turn > 0:
        task = '\nAct like the patient/family at an appointment with a geneticist. Using the above information, respond to the geneticist in one turn. Respond once, in up to 5 sentences. Do not continue the conversation past one response. Try to ask for information about a potential genetic diagnosis and treatment plan, if applicable.\n'
    else:
        task = '\nUsing the above information, pretend to be the patient or family at an appointment with a geneticist. The geneticist asks: [How can I help you today?]. Summarize your reason for visiting the geneticist in layman language that a non-medical person would use. Do not continue the conversation. Only respond as the patient or family, and respond to the question once: \nGeneticist: Hello, how can I help you today? \nPatient/Family: '


    total_string = './main -c 2048 -p "' + dialogue_current + task + '" -m ./llama-2-70b-chat/ggml-model-q4_0.gguf -s 0 > test.txt'
    os.system(total_string)
    a_file = open("test.txt", "r")
    data = a_file.read()
    prompts = data.split(task)
    patient_message = prompts[len(prompts)-1]
    patient_message.replace('"', '') #remove extra quotations within the response
    patient_message = 'Patient/Family: ' + patient_message
    dialogue['patient' + str(counter_turn)] = patient_message # append patient's turn to dialogue
    counter_turn = counter_turn + 1  
    print(counter_turn)



    print("geneticist1")
    vals = dialogue.values()
    result = list(vals)
    result.pop(0)
    dialogue_current = "\n".join(result)
    task = '\nAct like the geneticist. Using the information, respond only as the geneticist to the patient/family in one turn. Respond once, in up to 5 sentences. Do not continue the conversation past one response. Ask for more information about age, symptoms and family history. If you have enough information, try to provide the patient/family with a genetic diagnosis and treatment plan, but this is not necessary.\nGeneticist: '
    total_string = './main -c 2048 -p "' + dialogue_current + task + '" -m ./llama-2-70b-chat/ggml-model-q4_0.gguf -s 0 > test.txt'
    os.system(total_string)
    a_file = open("test.txt", "r")
    data = a_file.read()
    prompts = data.split(task)
    doctor_message = prompts[len(prompts)-1]
    doctor_message.replace('"', '') #remove extra quotations within the response
    doctor_message = 'Geneticist: ' + doctor_message
    dialogue['doctor' + str(counter_turn)] = doctor_message # append doctor's turn to dialogue
    # check if while-loop chat should end at some point
    counter_turn = counter_turn + 1
    print(counter_turn)


    print("patient2")
    vals = dialogue.values()
    dialogue_current = "\n".join(vals)
    task = '\nAct like the patient/family at an appointment with a geneticist. Using the above conversation and the patient vignette, answer the geneticists questions in detail. Try to ask for information about a potential genetic diagnosis and treatment plan, if applicable. Respond ONCE, in up to 8 sentences. Do not continue the conversation past one response. \nPatient/Family:'
    total_string = './main -c 2048 -p "' + dialogue_current + task + '" -m ./llama-2-70b-chat/ggml-model-q4_0.gguf -s 0 > test.txt'
    os.system(total_string)
    a_file = open("test.txt", "r")
    data = a_file.read()
    prompts = data.split(task)
    patient_message = prompts[len(prompts)-1]
    patient_message.replace('"', '') #remove extra quotations within the response
    patient_message = 'Patient/Family: ' + patient_message
    dialogue['patient' + str(counter_turn)] = patient_message # append patient's turn to dialogue
    counter_turn = counter_turn + 1  
    print(counter_turn)


    print("geneticist2")
    vals = dialogue.values()
    result = list(vals)
    result.pop(0)
    dialogue_current = "\n".join(result)
    task = '\nAct like the geneticist. Using the information, continue the conversation as the geneticist with the patient/family in one turn. Respond once, in up to 5 sentences. If you have enough information, try to provide the patient/family with a genetic diagnosis and treatment plan, but this is not necessary. Do not continue the conversation past the geneticist response.\nGeneticist: '
    total_string = './main -c 2048 -p "' + dialogue_current + task + '" -m ./llama-2-70b-chat/ggml-model-q4_0.gguf -s 0 > test.txt'
    os.system(total_string)
    a_file = open("test.txt", "r")
    data = a_file.read()
    prompts = data.split(task)
    doctor_message = prompts[len(prompts)-1]
    doctor_message.replace('"', '') #remove extra quotations within the response
    doctor_message = 'Geneticist: ' + doctor_message
    dialogue['doctor' + str(counter_turn)] = doctor_message # append doctor's turn to dialogue
    # check if while-loop chat should end at some point
    counter_turn = counter_turn + 1
    print(counter_turn)


    print("patient3")
    vals = dialogue.values()
    dialogue_current = "\n".join(vals)
    task = '\nAct like the patient/family at an appointment with a geneticist. Using the above information, respond to the geneticist in one turn. Respond once, in up to 5 sentences.Try to ask for information about a potential genetic diagnosis and treatment plan, if applicable. Do not continue the conversation past this response.\nPatient/Family:'
    total_string = './main -c 2048 -p "' + dialogue_current + task + '" -m ./llama-2-70b-chat/ggml-model-q4_0.gguf -s 0 > test.txt'
    os.system(total_string)
    a_file = open("test.txt", "r")
    data = a_file.read()
    prompts = data.split(task)
    patient_message = prompts[len(prompts)-1]
    patient_message.replace('"', '') #remove extra quotations within the response
    patient_message = 'Patient/Family: ' + patient_message
    dialogue['patient' + str(counter_turn)] = patient_message # append patient's turn to dialogue
    counter_turn = counter_turn + 1  
    print(counter_turn)


    print("geneticist3")
    vals = dialogue.values()
    result = list(vals)
    result.pop(0)
    dialogue_current = "\n".join(result)
    task = '\nAct like the geneticist. Using the information, continue the conversation as the geneticist with the patient/family in one turn. Respond once, in up to 5 sentences. If you have enough information, try to provide the patient/family with a genetic diagnosis and treatment plan, but this is not necessary. Do not continue the conversation past the geneticist response. \nGeneticist: '
    total_string = './main -c 2048 -p "' + dialogue_current + task + '" -m ./llama-2-70b-chat/ggml-model-q4_0.gguf -s 0 > test.txt'
    os.system(total_string)
    a_file = open("test.txt", "r")
    data = a_file.read()
    prompts = data.split(task)
    doctor_message = prompts[len(prompts)-1]
    doctor_message.replace('"', '') #remove extra quotations within the response
    doctor_message = 'Geneticist: ' + doctor_message
    dialogue['doctor' + str(counter_turn)] = doctor_message # append doctor's turn to dialogue
    # check if while-loop chat should end at some point
    counter_turn = counter_turn + 1
    print(counter_turn)


    filename = 'dialogue_update_' + str(counter) + '.txt'
    counter = counter + 1
    x = x + 1
    vals = dialogue.values()
    dialogue_current = "\n".join(vals)
    with open(filename, 'w') as f:
        f.write(dialogue_current)
    
