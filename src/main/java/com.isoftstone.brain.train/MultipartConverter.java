package com.isoftstone.brain.train;

import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.HttpMultipartMode;
import org.apache.http.entity.mime.MultipartEntityBuilder;

import java.io.File;

/**
 * Created by agibsonccc on 8/24/16.
 */
public class MultipartConverter implements Processor {
    /**
     * Processes the message exchange
     *
     * @param exchange the message exchange
     * @throws Exception if an internal processing error has occurred.
     */
    @Override
    public void process(Exchange exchange) throws Exception {
        // Read the incoming message…
        File file = exchange.getIn().getBody(File.class);
        String name = exchange.getIn().getHeader(Exchange.FILE_NAME, String.class);

        // Encode the file as a multipart entity…
        MultipartEntityBuilder entity = MultipartEntityBuilder.create();
        entity.setMode(HttpMultipartMode.BROWSER_COMPATIBLE);
        entity.setContentType(ContentType.MULTIPART_FORM_DATA);
        entity.addBinaryBody
                ("upfile", file, ContentType.MULTIPART_FORM_DATA, name);

        // Set multipart entity as the outgoing message’s body…
        exchange.getIn().setBody(entity.build());
    }
}
