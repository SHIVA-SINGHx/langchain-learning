from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


text = """
#Example-1
class Student(Person):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year

  def welcome(self):
    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)

#Example-2
class Student(Person):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year

x = Student("Mike", "Olsen", 2019)
"""

spiltter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 500,
    chunk_overlap= 0
)


result = spiltter.split_text(text)

print(len(result))
print(result[0])