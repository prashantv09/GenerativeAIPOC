import openai
import os

# your OpenAI API key
api_key = "sk-pxxxxxxxxxxxxx"
openai.api_key = api_key

def retrieve_data_from_workday():
    
    data = {
        "employee_name": "John Doe",
        "employee_id": "12345",
        "job_title": "Software Engineer",
        "salary": "$100,000",
        "hire_date": "2020-01-15",
    }
    return data

def ask_question_to_openai(question, context):
    response = openai.Completion.create(
        engine="text-davinci-002",  
        prompt=f"Context: {context}\nQuestion: {question}\nAnswer:",
        max_tokens=50,  
        temperature=0.7,  
    )
    return response.choices[0].text.strip()

def main():
   
    workday_data = retrieve_data_from_workday()

    
    context = f"Employee Name: {workday_data['employee_name']}\n" \
              f"Employee ID: {workday_data['employee_id']}\n" \
              f"Job Title: {workday_data['job_title']}\n" \
              f"Salary: {workday_data['salary']}\n" \
              f"Hire Date: {workday_data['hire_date']}"

    
    question = "What is the job title of the employee?"
    answer = ask_question_to_openai(question, context)
    print(f"Question: {question}\nAnswer: {answer}")

if __name__ == "__main__":
    main()