CC = gcc
CFLAGS = -std=c99
LDFLAGS = -lm
TARGET = gpt2

SRCS = gpt2.c
OBJS = $(SRCS:.c=.o)

.SILENT:
.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	clear
	./$(TARGET)
	rm -f $(OBJS) $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
