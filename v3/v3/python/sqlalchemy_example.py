try:
    from sqlalchemy import create_engine, Column, Integer, String
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
except ImportError:
    print("sqlalchemy not installed, skipping")
    exit(0)

engine = create_engine('sqlite:///:memory:')
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

session.add_all([User(name='Alice', age=30), User(name='Bob', age=25)])
session.commit()

users = session.query(User).all()
for user in users:
    print(f"{user.name} ({user.age})")
