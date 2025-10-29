#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


typedef struct {
  pthread_mutex_t writer;
  pthread_mutex_t reader;
  int data;
} queue;


void* printer(void* arg) {
  queue* q = (queue*) arg;
  while (1) {
    pthread_mutex_lock(&q->reader);
    printf("got data: %i \n", q->data);
    pthread_mutex_unlock(&q->writer);
  }
  return NULL;
}


int main(void) {
    queue q = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, 0};

    pthread_mutex_lock(&q.reader);
    pthread_t thread;
    pthread_create(&thread, NULL, printer, &q);

    printf("created thread?\n");


    for (int i = 0; i < 10; ++ i) {
      while (pthread_mutex_trylock(&q.writer));
      q.data = i;
      pthread_mutex_unlock(&q.reader);
    }

    while(pthread_mutex_trylock(&q.writer));
}
