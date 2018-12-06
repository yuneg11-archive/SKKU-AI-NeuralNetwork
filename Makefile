CPP		= g++ 
SRCS	= NNEBP.cpp
OBJS	= $(SRCS:.cpp=.o)
TARGET	= nn 
 
all : $(TARGET)

$(TARGET) : $(OBJS)
	$(CPP) $(OBJS) -o $@

.c.o:
	$(CPP) -c $< -o $@
 
clean :
	rm -f *o $(TARGET)