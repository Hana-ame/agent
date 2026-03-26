try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy import Column, Integer, String, select
    import asyncio
    SQLALCHEMY_ASYNC_AVAILABLE = True
except ImportError:
    SQLALCHEMY_ASYNC_AVAILABLE = False
    print("sqlalchemy[asyncio] not installed, skipping demo")
    exit(0)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

async def main():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:', echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        async with session.begin():
            session.add_all([User(name='Alice', age=30), User(name='Bob', age=25)])
        stmt = select(User)
        result = await session.execute(stmt)
        users = result.scalars().all()
        for u in users:
            print(f"{u.name} ({u.age})")

    await engine.dispose()

asyncio.run(main())
