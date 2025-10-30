#include <stdio.h>
#include "loader.c"

int main(int argc, char** argv) {
  uint64_t lines = 500;
  EventList es = load_file("data/mesy_dbg.csv", lines, true);

  printf("timestamp of event 100 is: %llu", es.events[100].time);
}
