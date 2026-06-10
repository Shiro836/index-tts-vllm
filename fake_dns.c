// fake_dns.c
#define _GNU_SOURCE
#include <netdb.h>
#include <arpa/inet.h>

int getnameinfo(const struct sockaddr *sa, socklen_t salen,
                char *host, socklen_t hostlen,
                char *serv, socklen_t servlen, int flags) {
    // always return dotted-decimal IP, skip DNS
    const struct sockaddr_in *sin = (const struct sockaddr_in*)sa;
    inet_ntop(AF_INET, &sin->sin_addr, host, hostlen);
    return 0;
}
